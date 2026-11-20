"""In-batch ranking that never treats a known eligible match as a negative."""
import torch
from sentence_transformers.base.losses.gradcache import CachedLossMixin, _create_minibatch
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss


class KnownPositiveRankingLoss(MultipleNegativesRankingLoss):
    """Standard query-to-cohort MNRL, masking other known-positive columns.

    Labels contain integer patient/cohort text identities. The complete positive
    pair set is sparse and uses only a few MiB; sampling and row coverage stay
    unchanged. Each row retains its designated positive on the diagonal.
    """
    def __init__(self, model, positive_keys, cohort_count, scale=20.0):
        super().__init__(model, scale=scale)
        if cohort_count < 1:
            raise ValueError('A positive cohort count is required')
        self.cohort_count = int(cohort_count)
        self.register_buffer('positive_keys', torch.as_tensor(positive_keys, dtype=torch.long,
            device=model.device).unique(sorted=True), persistent=False)
        if not len(self.positive_keys):
            raise ValueError('Known-positive masking requires positive pairs')

    def compute_loss_from_embeddings(self, embeddings, labels):
        if len(embeddings) != 2 or labels is None or labels.ndim != 2 or labels.shape[1] != 2:
            raise ValueError('Known-positive MNRL needs two text columns and patient/cohort identities')
        scores = self.similarity_fct(embeddings[0], embeddings[1]) * self.scale
        keys = labels[:, 0, None].long()*self.cohort_count + labels[None, :, 1].long()
        indices = torch.searchsorted(self.positive_keys, keys.contiguous())
        known = (indices < len(self.positive_keys)) & (
            self.positive_keys[indices.clamp(max=len(self.positive_keys)-1)] == keys)
        diagonal = torch.arange(len(labels), device=scores.device)
        if not bool(known[diagonal, diagonal].all()):
            raise ValueError('MNRL batch contains a pair outside the eligible training pairs')
        known[diagonal, diagonal] = False
        return torch.nn.functional.cross_entropy(scores.masked_fill(known, -torch.inf), diagonal)


def length_sorted_columns(sentence_features):
    """Sort inside the sampled batch and return inverse row permutations."""
    ordered, inverses = [], []
    for features in sentence_features:
        features = dict(features)
        mask = features.get('attention_mask')
        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            order = torch.argsort(mask.sum(dim=1), stable=True)
            inverse = torch.argsort(order)
            for key, value in features.items():
                if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(order):
                    features[key] = value.index_select(0, order)
        else:
            inverse = slice(None)
        ordered.append(features)
        inverses.append(inverse)
    return ordered, inverses


class CachedEmbeddingLoss(CachedLossMixin, torch.nn.Module):
    """Compute the complete ranking loss, replaying encoder chunks for gradients.

    Chunking never reduces the comparison set. Both MNRL's negative matrix and
    CoSENT's label-ordered comparisons see the entire optimizer batch. The shared
    SentenceTransformers machinery preserves dropout RNG and outer loss scaling;
    we also replay the forward autocast setting, including during backward.
    """
    def __init__(self, loss, mini_batch_size):
        super().__init__()
        if mini_batch_size < 1:
            raise ValueError('Embedding chunk size must be positive')
        self.loss = loss
        self.mini_batch_size = int(mini_batch_size)

    @property
    def model(self):
        return self.loss.model

    def calculate_loss(self, reps, labels=None, *, with_backward=False):
        labels, inverses = labels
        embeddings = [torch.cat(column)[inverse] for column, inverse in zip(reps, inverses)]
        loss = self.loss.compute_loss_from_embeddings(embeddings, labels)
        if with_backward:
            loss.backward()
            return loss.detach()
        return loss

    def embed_minibatch(self, sentence_feature, begin, end, with_grad,
                       copy_random_state, random_state=None):
        device_type = next(self.model.parameters()).device.type  # Also when DDP wraps the model.
        settings = (torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type))
        if random_state is not None:
            settings = random_state.trialspace_autocast
        with torch.autocast(device_type, enabled=settings[0], dtype=settings[1]):
            reps, state = super().embed_minibatch(sentence_feature, begin, end, with_grad,
                                                copy_random_state, random_state)
        if copy_random_state:
            state.trialspace_autocast = settings
        return reps, state

    def forward(self, sentence_features, labels=None):
        # Sort each text column independently inside this already-sampled batch
        # to avoid padding short texts to long ones in every encoder chunk.
        # Restore original row order before the loss, preserving every pairing.
        ordered, inverses = length_sorted_columns(sentence_features)
        return self.forward_cached(ordered, (labels, inverses))

    def get_config_dict(self):
        return dict(loss=self.loss.__class__.__name__, mini_batch_size=self.mini_batch_size,
                    sort_encoder_chunks_by_length=True)


class RetainedEmbeddingLoss(torch.nn.Module):
    """One encoder pass per text, retaining chunk graphs for the full-batch loss.

    Chunking reduces padding, not the comparison set or activation lifetime.
    Autograd keeps every chunk's graph until the ordinary loss backward pass.
    There are no detached embeddings, backward hooks or encoder recomputations.
    """
    uses_gradient_cache = False

    def __init__(self, loss, mini_batch_size=8, padded_token_budget=8192):
        super().__init__()
        if mini_batch_size < 1 or padded_token_budget < 1:
            raise ValueError('Encoder chunk size and padded-token budget must be positive')
        self.loss = loss
        self.mini_batch_size = int(mini_batch_size)
        self.padded_token_budget = int(padded_token_budget)

    @property
    def model(self):
        return self.loss.model

    def forward(self, sentence_features, labels=None):
        ordered, inverses = length_sorted_columns(sentence_features)
        embeddings = []
        for features, inverse in zip(ordered, inverses):
            mask = features.get('attention_mask')
            if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
                raise ValueError('Retained encoder chunks require padded text attention masks')
            # Last active position handles left padding correctly too. Reading
            # this small vector once avoids a CPU/GPU sync for every boundary.
            positions = torch.arange(1, mask.shape[1]+1, device=mask.device)
            widths = (mask.bool()*positions).amax(dim=1).tolist()
            begin, reps = 0, []
            while begin < len(widths):
                end, maximum = begin, 0
                while end < min(len(widths), begin+self.mini_batch_size):
                    candidate_max = max(maximum, widths[end])
                    if end > begin and candidate_max*(end-begin+1) > self.padded_token_budget:
                        break
                    maximum = candidate_max
                    end += 1
                chunk = _create_minibatch(features, begin, end)
                reps.append(self.model(chunk)['sentence_embedding'])
                begin = end
            embeddings.append(torch.cat(reps)[inverse])
        return self.loss.compute_loss_from_embeddings(embeddings, labels)

    def get_config_dict(self):
        return dict(loss=self.loss.__class__.__name__, mini_batch_size=self.mini_batch_size,
                    padded_token_budget=self.padded_token_budget, encoder_replay=False,
                    sort_encoder_chunks_by_length=True)

"""Full comparison-set gradients and encoder call counts for retained chunks."""
import copy
import importlib.metadata

from packaging.version import Version
import pytest

# trialspace_embedding_losses uses the sentence-transformers 6 module layout (pyproject pin).
if Version(importlib.metadata.version("sentence-transformers")) < Version("6.1"):
    pytest.skip("Requires sentence-transformers>=6.1", allow_module_level=True)

import torch
from sentence_transformers.sentence_transformer.losses import CoSENTLoss

from matchminer_ai.training.trialspace.losses import KnownPositiveRankingLoss, RetainedEmbeddingLoss


class Encoder(torch.nn.Module):
    device = torch.device('cpu')

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(40, 7)
        self.projection = torch.nn.Linear(7, 9)
        self.encoded_rows = 0

    def forward(self, features):
        self.encoded_rows += len(features['input_ids'])
        mask = features['attention_mask'].unsqueeze(-1)
        tokens = self.projection(self.embedding(features['input_ids']))
        return {'sentence_embedding': (tokens*mask).sum(1)/mask.sum(1)}


@pytest.mark.parametrize('kind', ['mnrl', 'cosent'])
@pytest.mark.parametrize('budget', [2, 9, 100])
def test_retained_chunks_preserve_loss_gradients_and_encode_each_text_once(kind, budget):
    torch.manual_seed(11)
    model = Encoder()
    reference = copy.deepcopy(model)
    features = [{'input_ids': torch.randint(1,40,(5,6)),
                 'attention_mask': torch.arange(6)[None,:] < torch.tensor(lengths)[:,None]}
                for lengths in [[6,1,4,2,3], [1,3,2,5,4]]]
    labels = torch.tensor([[0,0],[0,1],[1,1],[2,2],[3,1]])
    if kind == 'mnrl':
        loss = KnownPositiveRankingLoss(model, [0,1,4,8,10], 3)
        expected_loss = KnownPositiveRankingLoss(reference, [0,1,4,8,10], 3)
    else:
        loss, expected_loss = CoSENTLoss(model), CoSENTLoss(reference)
        labels = torch.tensor([0.,1.,.4,.8,-1.])
    actual = RetainedEmbeddingLoss(loss, mini_batch_size=2, padded_token_budget=budget)(features, labels)
    expected = expected_loss.compute_loss_from_embeddings(
        [reference(f)['sentence_embedding'] for f in features], labels)
    assert model.encoded_rows == 10
    (actual*.37).backward()
    assert model.encoded_rows == 10  # Backward must never call the encoder again.
    (expected*.37).backward()
    torch.testing.assert_close(actual, expected)
    for a,b in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-5)

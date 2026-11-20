#!/usr/bin/env python3
"""Stage embed: one side's key texts (patients or cohorts) encoded by one embedder, as training's mining encodes.

Embedders (common.EMBEDDERS): ``embeddinggemma2``, base google/embeddinggemma-2 at revision 914f7f89 (round-1
mining's encoder), and ``trialspace_round2``, the round-2 TrialSpace fit (the inference default). The encoder
settings are training's (trialspace.mining.encoder_settings: float32 weights, the trialspace prompt profile,
patients with the ``patient_summary`` prompt and cohorts with ``this_space``, mean pooling over the prompt, the
text tower, batches of 32) through the inference package's encoder, with the fit's and inference preset's
4,096-token cap. bf16 autocast applies on CUDA only, so CPU and CUDA vectors differ slightly: the identity names
the device kind, and retrieval refuses to compare vectors of different kinds. Texts are encoded in parts of
``--part-size`` (each part one encoder call) so an interrupted run resumes; ``vectors.npy`` is in input order.
"""
import argparse
from argparse import Namespace
import os
from pathlib import Path

from matchminer_ai.evaluation.retro_fidelity.common import EMBEDDERS, log, private, private_dir, read_jsonl, run_stage, stage, write_json  # noqa: E402

VERSION = 'retro-fidelity-embed-v1'
PROMPT = {'patients': 'patient_summary', 'cohorts': 'this_space'}


def settings(args):
    from matchminer_ai.training.trialspace.mining import encoder_settings
    model = EMBEDDERS[args.embedder]
    return encoder_settings(Namespace(model=model['model'], revision=model['revision'], cache_dir=model['cache_dir'],
                                      devices=(args.device,), encode_batch_size=args.batch_size,
                                      max_tokens=args.max_tokens))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, help='A patients or corpus stage directory')
    p.add_argument('--side', choices=sorted(PROMPT), required=True)
    p.add_argument('--embedder', choices=sorted(EMBEDDERS), required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=32, help='torch CPU threads')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--part-size', type=int, default=1024)
    p.add_argument('--output', required=True)
    stage.add_check_arg(p)
    args = p.parse_args(argv)
    private()
    from matchminer_ai.embedding.encoder import encoder_identity
    config = settings(args)
    source = stage.receipt_id(args.input)
    identity = dict(version=VERSION, side=args.side, embedder=args.embedder, input=source,
                    encoder=encoder_identity(config), device_kind='cuda' if args.device.startswith('cuda') else 'cpu',
                    part_size=args.part_size)
    output = Path(args.output)

    def work():
        import numpy as np
        import torch
        torch.set_num_threads(args.threads)
        from matchminer_ai.training.trialspace.finetune import require_embeddinggemma2
        from matchminer_ai.embedding.encoder import open_encoder
        rows = read_jsonl(Path(args.input) / f'{args.side}.jsonl')
        texts = [row['text'] for row in rows]
        parts = private_dir(output / 'parts')
        require_embeddinggemma2(config.model, revision=config.revision, cache_dir=config.cache_dir)
        starts = list(range(0, len(texts), args.part_size))
        with open_encoder(config) as encoder:
            tokenizer = encoder.model.tokenizer
            prompt = encoder.model.prompts[PROMPT[args.side]]
            lengths = [len(ids) for ids in tokenizer([prompt + t for t in texts], add_special_tokens=True)['input_ids']]
            for number, start in enumerate(starts):
                path = parts / f'{number:05d}.npy'
                if path.exists():
                    continue
                values = encoder.encode(texts[start:start + args.part_size], PROMPT[args.side])
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.save(stream, values, allow_pickle=False)
                temporary.replace(path)
                log(f'embed {args.embedder}/{args.side}: {min(start + args.part_size, len(texts))}/{len(texts)}')
        vectors = np.concatenate([np.load(parts / f'{n:05d}.npy') for n in range(len(starts))])
        if vectors.shape[0] != len(rows) or not np.isfinite(vectors).all() or \
                not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
            raise ValueError('Invalid embeddings')
        with (output / 'vectors.tmp').open('wb') as stream:
            np.save(stream, vectors.astype(np.float32), allow_pickle=False)
        (output / 'vectors.tmp').replace(output / 'vectors.npy')
        summary = dict(rows=len(rows), dimension=int(vectors.shape[1]), device=args.device,
                       ids_sha256=stage.digest([row['id'] for row in rows]),
                       tokens=dict(max=max(lengths), over_cap=sum(n > args.max_tokens for n in lengths),
                                   median=sorted(lengths)[len(lengths) // 2]))
        write_json(output / 'embedding.json', summary)
        for path in parts.iterdir():
            path.unlink()
        parts.rmdir()
        return summary

    run_stage(output, 'retro-embed', identity, ['vectors.npy', 'embedding.json'], work, check=args.check,
              upstream=dict(input=source))


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()

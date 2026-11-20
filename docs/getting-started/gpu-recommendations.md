# GPU recommendations

GPU needs differ by step. The LLM is by far the largest model and is the only one that can run on another machine.
TrialSpace and the decision model always run in-process on the machine running `matchminer-ai`. This page covers
model sizes, memory settings, the software stack and the hardware we use.

## Models

| Model | Default | Size | GPU memory |
| --- | --- | --- | --- |
| LLM | [`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) | 120B parameters, 12B active per token | About 70 GB of NVFP4 weights, plus KV cache |
| TrialSpace | Your checkpoint, fine-tuned from [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) | About 270M (text tower only) | About 1 GB of float32 weights, plus activations |
| Decision model | [`LiquidAI/d1-2B`](https://huggingface.co/LiquidAI/d1-2B), or a fine-tuned checkpoint | About 1.9B | About 3.8 GB of bfloat16 weights, plus activations |

- **LLM:** Nemotron 3.5 Super is a hybrid Mamba-Transformer mixture of experts, served text only
  (`language_model_only`). Only 8 of its 88 layers are attention layers, so long contexts need little KV-cache memory.
  The NVFP4 build needs native FP4 kernels, which means a Blackwell GPU. The preset runs it on one GPU per engine
  (`tensor_parallel_size: 1`, `gpu_memory_utilization: 0.9`), with a 100,000-token context for trials and 200,000 for
  patients.
- **Without Blackwell:** select the FP8 build,
  [`RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8`](https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8),
  by setting each LLM section's `model_name`. In-process, also raise `local.engine.tensor_parallel_size` to fit it
  across GPUs. It gets the same sampling profile. Or run the LLM on an OpenAI-compatible server elsewhere; see
  [choosing an inference setup](../user-guide/local-vs-remote.md).
- **Other LLMs:** smaller models with a registered profile, such as Gemma 4, are selected the same way; see
  [LLM runtime](../reference/llm-runtime.md). The preset's token budgets and settings were chosen for Nemotron.
- **TrialSpace:** float32 weights with bfloat16 autocast on CUDA. `embedding.batch_size` (32) and
  `embedding.max_seq_length` (4,096 tokens) bound activation memory.
- **Decision model:** bfloat16. Activations scale with `decision_model.token_budget`, the padded tokens per forward
  pass (65,536). Lower it if the GPU runs out of memory.

### Sharing GPUs

An in-process vLLM engine reserves `gpu_memory_utilization` (90%) of each GPU it uses for as long as it stays loaded.
TrialSpace and the decision model cannot load on that GPU at the same time. Either:

- put them on another GPU (`embedding.device: cuda:1`, `decision_model.device: cuda:1`);
- release the engine first (`matchminer_ai.llm.backends.clear_local_llm_cache()`), or run the LLM steps in a separate
  process;
- or run the LLM on a server.

## Estimating memory

Model weights take the number of parameters times the bytes per parameter of their precision. Leave room on top for
the KV cache, activations and the framework. NVIDIA's
[GPU Memory Essentials for AI Performance](https://developer.nvidia.com/blog/gpu-memory-essentials-for-ai-performance/)
discusses the overheads.

| Precision | Bytes per parameter |
| --- | --- |
| FP32 | 4 |
| BF16, FP16 | 2 |
| FP8, INT8 | 1 |
| FP4, INT4 | 0.5 |

For example, Nemotron 3.5 Super has 120 billion parameters: about 60 GB at FP4 and about 120 GB at FP8. Its NVFP4
checkpoint is about 70 GB on disk.

## Software stack

The package's GPU-dependent requirements (see `pyproject.toml`):

- `torch>=2.11.0`
- `vllm>=0.20.2`
- `transformers>=5.19.0`
- `sentence-transformers>=6.1`, which saved the TrialSpace checkpoints

For Nemotron 3.5 Super, NVIDIA recommends vLLM 0.32.0 or its documented nightly. Install a PyTorch build that
matches your CUDA driver; see the [PyTorch](https://pytorch.org/get-started/locally/) and
[vLLM](https://docs.vllm.ai/) installation guides.

## Our experience

| GPU | How we run the LLM |
| --- | --- |
| NVIDIA GB300, RTX Pro 6000 (96 GB) | Nemotron 3.5 Super NVFP4, one vLLM server per GPU at tensor parallel size 1 |
| NVIDIA H100 NVL (94 GB), without native FP4 | Nemotron 3.5 Super FP8, one vLLM server across four GPUs |

TrialSpace and the decision model each use one GPU.

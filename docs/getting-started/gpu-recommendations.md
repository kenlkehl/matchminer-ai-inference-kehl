# GPU Recommendations

Determining the type of GPUs needed to run an AI pipeline can be complicated.  The minimum GPU and compute requirements for matchminer-ai-inference also vary for each step. While we cannot summarize all factors which impact AI performance, this guide provides information on the following:

1. [Models used in our pipeline](#models-used-in-our-pipeline);
2. [Estimating Minimum GPU memory for a LLM model](#minimum-gpu-memory);
3. [Minimum requirements for the software stack](#minimum-software-stack-requirements); and
4. [Our experience](#our-experience).

## Models Used In Our Pipeline
MatchMiner-AI is a pipeline which uses 3 different models:

- An LLM for patient summarization and trial 'space' extraction. The default is NVIDIA's Nemotron 3.5 Super, [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) (a 120B-parameter hybrid Mamba-Transformer mixture of experts with 12B active parameters, served text only). Earlier releases used variants of Google's Gemma-4-31B-IT, which remains supported.
- 'TrialSpace' - an embedding model. The default is, for now and on purpose, a local unpublished checkpoint: the n=5000 training run's round-2 TrialSpace, fine-tuned from Google's EmbeddingGemma 2 [https://huggingface.co/google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (about 270M parameters; float32 weights with bfloat16 autocast take about 1 GB of GPU memory plus activations). The published release [https://huggingface.co/ksg-dfci/TrialSpace-0526](https://huggingface.co/ksg-dfci/TrialSpace-0526) was fine-tuned from Qwen 3 0.6B Embedding [https://huggingface.co/Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
- The decision model - a 1.9B-parameter System One model that answers the match, boilerplate and per-criterion yes/no questions: [https://huggingface.co/LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B) by default (LFM Open License v1.0), or a checkpoint fine-tuned from it by the training repository. In bfloat16 it needs a few GB of GPU memory plus activations for `decision_model.token_budget` (65,536) tokens per forward pass.

For any given model, minimum compute/GPU requirements could be determined from others' experiences either with the model itself (e.g. Nemotron 3.5 Super or Gemma-4-31B-IT) or with the base (pre-trained) models (EmbeddingGemma 2, d1-2B).

## Minimum GPU Memory
Nvidia provides some guidance and a short-hand metric for determining how much GPU VRAM is required to run a given LLM. (See [GPU Memory Essentials for AI Performance](https://developer.nvidia.com/blog/gpu-memory-essentials-for-ai-performance/)).  Specifically, it recommends having at least Parameter * Precision * 2 VRAM. Where Parameter is the number of parameters in the model and Precision is the bytes per parameter for the model's precision format -- FP32, FP16, FP8 or FP4.  (See conversion table below.)

Precision Bytes per Parameter. From [GPU Memory Essentials for AI Performance](https://developer.nvidia.com/blog/gpu-memory-essentials-for-ai-performance/)

| Precision | Bytes per Parameter |
|-----------|----------------------|
| INT32/FP32 | 4 |
| INT16/FP16 | 2 |
| INT8/FP8 | 1 |
| INT4/FP4 | 0.5 |

For example, when we run [RedHatAI/gemma-4-31B-it-FP8-Dynamic](https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-Dynamic) for patient and trial summarization, we should have at least

*31,000,000,000 parameter* (31 billion parameters) * *1 byte/parameter* * *2 VRAM* (~62 GB)

In practice, we run RedHatAI/gemma-4-31B-it-FP8-Dynamic on H100 (NVL PCIe) GPUs with 94GB VRAM each.

Nemotron 3.5 Super's NVFP4 build needs native FP4 kernels (a Blackwell GPU). We serve it with tensor parallel size 1, one server per GPU, on GB300 and RTX Pro 6000 (96 GB) GPUs; only 8 of its 88 layers are attention layers, so long contexts need little KV-cache memory. On H100s, which lack native FP4, we serve the FP8 build, [RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8](https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8), across four H100 NVL 96 GB GPUs; select it by setting each section's `model_name`.

## Minimum Software Stack Requirements
The [dependencies in our pipeline](https://github.com/dfci/matchminer-ai-inference/blob/main/pyproject.toml) include packages which have particular compute/GPU requirments.  Currently these include:

- "torch>=2.11.0"
- "vllm>=0.20.2"

Documentation for each package is available online:
- [PyTorch (torch)](https://pytorch.org/get-started/locally/)
- [vllm](https://docs.vllm.ai/en/v0.6.5/getting_started/installation.html)

Other software such as sentence-transformers and transformers are dependent on the torch version.

## Our Experience
We have successfully run the entire pipeline on the following machines:

| Machine | Notes |
|---------|-------|
| RTX Pro 6000 (available on the cloud) | Has native FP4 kernels, so can run NVFP4 versions of Gemma 4, such as nvidia/Gemma-4-31B-IT-NVFP4. |
| H100 NVL PCI / 300W, which have native FP8 but not native FP4 | Can run RedHatAI/gemma-4-31B-it-FP8-Dynamic, albeit slowly. (See comment below.) |
| NVIDIA A100 80GB | We have run the example notebook steps with google/gemma-4-31B-it as the LLM on this machine type.|

RedHatAI/gemma-4-31B-it-FP8-Dynamic at the context lengths we need runs slowly on our H100 machines. It runs faster as an NVFP4 quant (eg [https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) ) on a Blackwell GPU, eg, the RTX Pro 6000.

# DiffuGuard

Official repo for our paper “DiffuGuard: How Intrinsic Safety is Lost and Found in Diffusion Large Language Models”.

## Initial Setup

Create a virtual environment and install dependencies.

```shell
conda create -n diffuguard python==3.11
conda activate diffuguard
pip install -r requirements.txt
```

Download dLLMs locally (recommended for stability) and keep paths handy.

```shell
bash hf_models/model_download.sh
```

Optionally set evaluator credentials if you plan to compute ASR with the OpenAI scripts.

```shell
# for analysis/evaluation.py
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=your_url
```

## Core Experiments (Analysis)

- Logits heatmap (Figure 2):
  - `python analysis/heatmap.py` (edit `MODEL_PATH` and dataset in the file if needed)
- Random remasking (Figure 3):
  - `bash analysis/exp_remask_randomness.sh`
- Token injection (Figures 4 & 5):
  - `bash analysis/exp_token_injection.sh`
- Batch evaluation of results:
  - `bash analysis/eval.sh`

## Quick Start

Run DiffuGuard on LLaDA with PAD attack, hidden audit, and repair:

```shell
python models/jailbreakbench_llada.py \
  --model_path hf_models/LLaDA-8B-Instruct \
  --attack_method PAD \
  --attack_prompt path/to/prompts.json \
  --output_json out_llada.json \
  --steps 64 --gen_length 128 --block_length 128 \
  --sp_mode hidden --sp_threshold 0.35 \
  --refinement_steps 8 --remask_ratio 0.9
```

Dream and MMaDA runners have similar usage:

```shell
python models/jailbreakbench_dream.py --model_path hf_models/Dream-v0-Instruct-7B --attack_method pad --attack_prompt path/to/prompts.json --output_json out_dream.json --gen_length 128 --steps 64 --mask_counts 36 --sp_mode hidden --sp_threshold 0.35 --refinement_steps 8 --remask_ratio 0.9
python models/jailbreakbench_mmada.py --model_path hf_models/MMaDA-8B-MixCoT --attack_method PAD --attack_prompt path/to/prompts.json --output_json out_mmada.json --steps 64 --sp_mode hidden --sp_threshold 0.35 --refinement_steps 8 --remask_ratio 0.9
```

## Key Hyperparameters

- `steps`, `gen_length`, `block_length`: diffusion steps and decoding span
- `remasking`: `off | low_confidence | adaptive_step` (annealed randomness)
- `sp_mode`: `off | hidden`; `sp_threshold`
- `refinement_steps` (e.g., 8), `remask_ratio` (e.g., 0.9)

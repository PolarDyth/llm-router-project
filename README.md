# LLM Router

A homebrew cross-encoder that scores `(user prompt, model description)` pairs
and picks the LLM best suited to a given prompt.

Runs on **Apple Silicon (MPS)**, **NVIDIA GPUs (CUDA)**, and **AMD GPUs**
(ROCm on Linux/Windows, DirectML on Windows) — plus CPU as a fallback.

## Project layout

```
data/                                # gitignored: datasets + model_profiles.json
  model_profiles.json                # scraped OpenRouter profiles (input)
  router_training_pairs_scrubbed.csv # built by build_scrubbed_arena.py
  router_training_pairs_v2.csv       # built by build_better_dataset.py
models/dynamic_router/final/         # trained cross-encoder (output)
device_utils.py                      # shared accelerator detection
scripts/                             # dataset builders + inference demo
training/train_router.py             # training entry point
```

## Setup

1. Create a virtualenv and install PyTorch **for your accelerator**
   (only macOS can skip this step — the plain PyPI wheel already uses MPS):

   | Platform | PyTorch install |
   |---|---|
   | macOS (Apple Silicon) | just run step 2 |
   | NVIDIA (Linux) | PyPI wheels already bundle CUDA: just run step 2, or pick a specific CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) |
   | NVIDIA (Windows) | `pip install torch --index-url https://download.pytorch.org/whl/cu126` (the default PyPI Windows wheel is CPU-only) |
   | AMD (Linux, ROCm) | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.4` — see [pytorch.org](https://pytorch.org/get-started/locally/) for the current tag |
   | AMD (Windows, RX 7000/9000) | AMD's official ROCm-on-Windows wheels: <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html> |
   | AMD (Windows, older cards) | `pip install torch-directml` after step 2 (DirectML fallback; the only option for e.g. RX 6000-series) |

2. Install the remaining dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Device detection happens automatically at startup (`cuda` → `mps` →
`directml` → `cpu`) and the chosen device is printed. Every script accepts
`--device` (train) or auto-detects (test) to force a specific one.

## Usage

```bash
# 1. Put scraped OpenRouter profiles at data/model_profiles.json
#    (schema: {"<org/model-id>": {"name": ..., "description": ...}})

# 2. Build a training dataset (either one)
python scripts/build_scrubbed_arena.py    # from lmarena-ai/arena-human-preference-140k
python scripts/build_better_dataset.py    # synthetic from keyword matching

# 3. Train (auto-picks the newest dataset CSV in data/)
python training/train_router.py
python training/train_router.py --epochs 5 --device cuda   # override anything

# 4. Try it
python scripts/test_router.py
```

Training reproduces the original recipe (AdamW lr 2e-5, linear warmup 100
steps, batch 32, 3 epochs) with a plain PyTorch loop so it runs identically
on every accelerator above. See `python training/train_router.py --help` for
all knobs.

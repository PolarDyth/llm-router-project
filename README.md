# LLM Router

A homebrew LLM router. The current approach is **complexity-based**: datasets
of (user prompt, model picked) are labeled by translating every model name
into a **complexity tier 1–5**, so the router learns how much model a prompt
needs instead of memorizing model names.

Runs on **Apple Silicon (MPS)**, **NVIDIA GPUs (CUDA)**, and **AMD GPUs**
(ROCm on Linux/Windows, DirectML on Windows) — plus CPU as a fallback.

## Project layout

```
data/model_registry/
  models.json                        # generated registry: canonical chat LLMs -> complexity 1-5 (checked in)
  overrides.yaml                     # hand-maintained tier fixes + pruned 2023-era models (checked in)
  cache/                             # gitignored: raw source catalogs
  build_report.md                    # gitignored: per-build disagreements + review list
router/model_registry.py             # loader + resolve(name) -> entry, shared name normalization
scripts/build_model_registry.py      # merges 4 catalogs, scores tiers, emits models.json
scripts/audit_model_coverage.py      # % of a dataset's model names the registry resolves
models/dynamic_router/final/         # trained cross-encoder (output)
device_utils.py                      # shared accelerator detection
scripts/ training/ notebooks/        # dataset builders + training entry point
eval_reports/                        # gitignored: evaluation + coverage reports
```

## Model complexity registry

`data/model_registry/models.json` maps ~3,500 canonical chat LLMs (2022–2026)
to an **absolute capability tier** — Claude Opus 5.5 = 5, a flash/mini model
= 1, regardless of release year. Each entry carries canonical id, display
name, all known aliases, release date, the raw signals, and which signal
decided the tier (`basis`).

Tier scale:

| Tier | Class | Examples |
|---|---|---|
| 1 | tiny/flash | gemma-2-9b, gemini-2.0-flash-lite, vicuna-7b |
| 2 | light/mini | gpt-4o-mini, gemini-2.0-flash, claude-3-haiku, gpt-3.5-turbo |
| 3 | mid/pro | llama-3.1-70b, deepseek-chat, gemini-1.0-pro, claude-2 |
| 4 | pro / 2024-frontier | gpt-4o, claude-3-5-sonnet, o1-mini, gemini-1.5-pro |
| 5 | frontier | o1, claude-opus-4, kimi-k3.1-pro, claude-opus-5.5 |

Tiers are scored by `scripts/build_model_registry.py`: a log-price prior
(blended $/1M tokens), adjusted one step toward quality when the
Artificial-Analysis intelligence index disagrees, with parameter-count
fallback for unpriced open models, era caps (nothing before gpt-4 outranks
tier 3, nothing before o1 outranks tier 4), and `overrides.yaml` winning over
everything. Sources: the [LiteLLM catalog](https://github.com/BerriAI/litellm)
(pricing, historical models, alias variants), the
[OpenRouter models API](https://openrouter.ai/docs#models) (pricing +
quality benchmarks), [models.dev](https://models.dev) (canonical ids, release
dates), and [Epoch AI's model database](https://epoch.ai/data/ai-models)
(parameters for 2022–2023 fallback).

Usage:

```python
from router.model_registry import get_registry
registry = get_registry()
registry.resolve("claude-3-opus-20240229")   # -> full entry
registry.complexity("gpt-4-1106-preview")    # -> 4  (via openai/gpt-4-turbo)
```

Regenerate after editing `overrides.yaml` (or pass `--refresh` to re-download
source catalogs), then re-audit any dataset the router will train on:

```bash
python scripts/build_model_registry.py
python scripts/audit_model_coverage.py                       # arena-human-preference-140k
python scripts/audit_model_coverage.py --dataset <hf_id>     # any dataset with model columns
python scripts/audit_model_coverage.py --file data/foo.csv --columns model,model_id
```

Coverage target is ≥ 99% of dataset rows; both arena datasets
(`arena-human-preference-140k`, `lmsys-chatbot_arena_conversations`) audit at
100%.

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
python scripts/build_capability_dataset.py  # capability archetypes (default)
python scripts/build_scrubbed_arena.py      # from lmarena-ai/arena-human-preference-140k

# 3. Train (auto-picks the newest dataset CSV in data/)
python training/train_router.py
python training/train_router.py --epochs 5 --device cuda   # override anything
# A --holdout-frac (default 5%) of unique prompts is held out for a
# per-epoch validation loss.

# 4. Try it
python scripts/test_router.py "explain how airline pricing works"
python scripts/test_router.py --interactive   # REPL: type prompts, get rankings
```

Training reproduces the original recipe (AdamW lr 2e-5, linear warmup 100
steps, batch 32, 3 epochs) with a plain PyTorch loop so it runs identically
on every accelerator above. See `python training/train_router.py --help` for
all knobs.

## Testing the router

Three steps, all local (inference only — fine on a Mac):

```bash
# 1. Fetch today's OpenRouter catalog (the "current data" side):
#    data/model_profiles_live.json + a diff vs the training snapshot
python scripts/fetch_catalog.py

# 2. Build the held-out eval sets (prompts guaranteed absent from training):
#    data/eval/arena_holdout.jsonl (real battles) + current_bench.jsonl
#    (hand-authored current-world prompts, eval_lib/current_bench.py)
python scripts/build_eval_set.py

# 3. Run the harness — writes eval_reports/<timestamp>/{report.md,details.csv,
#    pairwise.csv,confusion.csv} and prints headline numbers (~2-3 min on MPS)
python scripts/evaluate_router.py
```

What the harness measures:

- **Pairwise** — real arena battles held out of training: does the winner's
  description outscore the loser's? (The capability-trained checkpoint doesn't
  optimize this signal; read it as a diagnostic floor.)
- **Routing** — the current bench: does a top-ranked model's archetype match
  the prompt's expected archetype? Compared across three description modes:
  `raw` (OpenRouter blurb as-is), `scrubbed` (exactly the training pipeline),
  and `enriched` (scrubbed + a name-free capability line built from live
  metadata: context window, input modalities, tool/reasoning support, price
  tier) — so every candidate is described in decent depth without leaking
  brand names.
- **Novelty** — same metrics restricted to models added to the catalog since
  the training snapshot: descriptions the router has never seen.

Point `--prompts any.jsonl` at your own prompts (optional `archetype` field
per line adds them to the metrics). `scripts/test_router.py` uses the same
catalog/mode machinery for ad-hoc routing (`--catalog live|snapshot`,
`--mode raw|scrubbed|enriched`, marks models new since training with `NEW`).

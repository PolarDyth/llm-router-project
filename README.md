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
  overrides.yaml                     # hand-maintained tier fixes + created entries (checked in)
  cache/                             # gitignored: raw source catalogs
  build_report.md                    # gitignored: per-build disagreements + review list
router/model_registry.py             # loader + resolve(name) -> entry, shared name normalization
scripts/build_model_registry.py      # merges 4 catalogs, scores tiers, emits models.json
scripts/audit_model_coverage.py      # % of a dataset's model names the registry resolves
scripts/build_training_dataset.py    # builds the (prompt -> complexity) dataset from arena + RouterBench
device_utils.py                      # shared accelerator detection
notebooks/train_router_colab.ipynb   # the only training environment (Colab T4)
data/router_training_dataset.jsonl   # gitignored: generated training data
eval_reports/                        # gitignored: coverage + build reports
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
python scripts/audit_model_coverage.py --all                  # full training suite (4 arena sets + RouterBench)
python scripts/audit_model_coverage.py                        # arena-human-preference-140k only
python scripts/audit_model_coverage.py --dataset <hf_id>     # any dataset with model columns
python scripts/audit_model_coverage.py --file data/foo.csv --columns model,model_id
```

Coverage target is ≥ 99% of rows per source. `lmsys/chatbot_arena_conversations`
(33k) is gated on the Hub: accept its license on the dataset page and run
`hf auth login` to include it.

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

Device detection (`cuda` → `mps` → `directml` → `cpu`) lives in
`device_utils.py` and is shared by anything that runs on a local
accelerator; DirectML loads lazily via importlib so it is never a hard
dependency.

## Training data

`scripts/build_training_dataset.py` builds `data/router_training_dataset.jsonl`
(gitignored, regenerable) from public datasets whose rows pair a **prompt
with one or more model names** — each name is translated into a complexity
tier via the registry, so every row becomes (prompt → how much model it
needed). Battles cover mid-2023 → mid-2025, so the tier signal stays
era-consistent with the registry.

| source | rows | shape | label_kind |
|---|---|---|---|
| `lmarena-ai/arena-human-preference-140k` | ~140k battles | prompt + 2 models + human winner (CC-BY-4.0) | `winner_tier` |
| `lmarena-ai/arena-human-preference-100k` | ~100k battles | " | `winner_tier` |
| `lmarena-ai/arena-human-preference-55k` | ~55k battles | " (apache-2.0) | `winner_tier` |
| `lmsys/chatbot_arena_conversations` | ~33k battles | " (CC-BY-NC-4.0, gated) | `winner_tier` |
| `withmartian/routerbench` (0-shot) | ~30k prompts × 11 models | prompt + per-model quality + cost | `min_solving_tier` |

Two label kinds:

- **`winner_tier`** — the tier of the model that won the battle. Ties and
  both-bad rows are kept with `label_tier: null` (they still carry
  `tier_a`/`tier_b`).
- **`min_solving_tier`** — RouterBench only: the cheapest tier whose best
  model reaches the quality threshold (per-prompt best − ε, or
  exact-correct when scores are binary). These prompts are exam-style
  (MMLU, GSM8K, MBPP, …), so treat them as an auxiliary signal alongside
  the organic arena prompts.

Every row also keeps the raw ingredients (`tier_a`, `tier_b`, `winner` or
`candidate_tiers`), so the notebook can choose any training objective —
per-prompt tier classification, winner-vs-loser contrast — without
rebuilding the data.

`scripts/build_training_pairs.py` derives the training-ready form from that
file (local, no network): one row per **(prompt, tier, outcome, source)**
sample for pairwise training — the model learns whether a tier *wins,
loses, or ties* on a prompt, then routes by scoring tiers 1–5 and taking
the argmax. Decisive battles emit (winner tier, 1.0) + (loser tier, 0.0);
ties/both-bad emit both tiers at 0.5; RouterBench emits its solving tiers
at 1.0 and the rest at 0.0. Decisive same-tier battles are skipped (no
complexity gradient; `--keep-same-tier` to include). `--max-pairs N` caps
the output with a seeded, proportionally **stratified sample over
(source, tier)** — never truncate the file by head, rows are ordered by
source era.

```bash
python scripts/build_training_dataset.py                       # arena suite
python scripts/build_training_dataset.py --sources arena,routerbench
python scripts/build_training_dataset.py --limit 500           # smoke run
python scripts/build_training_pairs.py                         # full: ~534k pairs
python scripts/build_training_pairs.py --max-pairs 15000       # capped for quick training
```

Gates: per-source model coverage must be ≥ 99% (unmapped names go to
`overrides.yaml`, rebuild the registry, retry) and duplicate battles (same
prompt + model pair) across releases are dropped. The build writes
`eval_reports/training_dataset_report.md` — per-source counts, label-tier
histograms, prompt-length stats; eyeball it after every build.

## Training

Training runs in **Colab (T4) only** — the local machine never trains.
`notebooks/train_router_colab.ipynb` is self-contained: upload
`data/router_training_pairs.jsonl` (built locally by
`scripts/build_training_pairs.py --max-pairs 15000`), and it fine-tunes a
MiniLM cross-encoder as a **pairwise tier scorer** — the model reads
(prompt, tier) and predicts win/lose/tie, using only the `prompt`, `tier`
and `outcome` fields (the `source` column is ignored). Routing = argmax
over the five fixed tier descriptions (saved alongside the weights as
`tier_texts.json`). Manual PyTorch loop, no `pin_memory`, DirectML loaded
lazily — every accelerator path stays open. The retired cross-encoder
training pipeline was removed from `training/` (recoverable from git
history, `993610e`).

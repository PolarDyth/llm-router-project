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
router/                              # the domain: one module per pipeline stage
  contracts.py                       # (prompt, tier) interface: TIERS, TIER_TEXTS, MAX_LEN, VAL_PROMPT_MOD
  model_registry.py                  # loader + resolve(name) -> entry, shared name normalization
  catalogs.py                        # 4 vendor catalogs -> merged canonical groups
  scoring.py                         # groups -> complexity tier (price/quality/era rules, overrides)
  sources.py                         # the dataset source catalog, shared by audit + build
  dataset.py                         # source rows -> labeled dataset rows + the coverage gate
  pairs.py                           # dataset rows -> (prompt, tier, outcome) pairs
  metrics.py                         # prompt-hash split + pairwise/routing/majority math
  devices.py                         # cross-platform accelerator selection
scripts/                             # thin CLIs: args, stage ordering, markdown reports
  build_model_registry.py            # merges 4 catalogs, scores tiers, emits models.json
  audit_model_coverage.py            # % of a dataset's model names the registry resolves
  build_training_dataset.py          # builds the (prompt -> complexity) dataset
  build_training_pairs.py            # derives the pairwise training file
  eval_checkpoint.py                 # local inference-only eval of a trained checkpoint
tests/                               # pytest suite: stage rules, scoring invariants,
                                     # notebook contract, pre-restructure byte fingerprints
notebooks/train_router_colab.ipynb   # the only training environment (Colab T4)
data/router_training_dataset.jsonl   # gitignored: generated training data
eval_reports/                        # gitignored: coverage + build reports
```

`router/` is an installed package (`pip install -e .`), so the scripts import it
plainly — there is no `sys.path` juggling. Rules live in the stage modules; a
script that grows a rule has the rule in the wrong place.

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

Tiers are scored by `router/scoring.py` (invoked by
`scripts/build_model_registry.py`): a log-price prior
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

Pass `--out`/`--report`/`--cache` to `build_model_registry.py` to build a
candidate registry somewhere other than the checked-in one and diff it before
committing. `--all` audits `router.sources.AUDITABLE` — the same source catalog
the dataset build reads and gates against, so the audit predicts whether a build
will pass rather than merely resembling the gate.

Coverage target is ≥ 99% of rows per source that carries model names. `lmsys/chatbot_arena_conversations`
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

3. Install the `router` package in editable mode (required — the scripts import
   it), with pytest for the test suite:

   ```bash
   pip install -e ".[dev]" --no-deps
   ```

   `--no-deps` stops pip from re-resolving the accelerator-specific torch build
   you picked in step 1.

Device detection (`cuda` → `mps` → `directml` → `cpu`) lives in
`router/devices.py` and is what every repo-local script imports; the Colab
notebook carries its own three-line copy because Colab has no checkout. DirectML
loads lazily via importlib so it is never a hard dependency.

Run the suite with `python -m pytest -q` (~3 s, no network). Tests that need the
downloaded vendor catalogs skip themselves until step 4 of the registry section
below has been run.

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
| `RowRed/ComplexityRouter` (train split) | 4k OASST2 prompts | prompt + complexity level 0–3, no model names | `annotated_level` |

Three label kinds:

- **`winner_tier`** — the tier of the model that won the battle. Ties and
  both-bad rows are kept with `label_tier: null` (they still carry
  `tier_a`/`tier_b`).
- **`min_solving_tier`** — RouterBench: the cheapest tier whose best
  model reaches the quality threshold (per-prompt best − ε, or
  exact-correct when scores are binary). These prompts are exam-style
  (MMLU, GSM8K, MBPP, …), so treat them as an auxiliary signal alongside
  the organic arena prompts.
- **`annotated_level`** — ComplexityRouter: complexity levels assigned directly
  by Qwen3.5-4B, mapped linearly onto tiers 1–4 (`LEVEL_TO_TIER`), so nothing is
  resolved through the registry and the coverage gate does not apply. The row
  reuses the RouterBench shape with `solving_tiers=[label_tier]` — gold tier
  wins, the other four lose. 813 rows are synthetic rewrites, flagged
  `synthetic: true`. Tier 5 stays arena/RouterBench-only by construction.

Every row also keeps the raw ingredients (`tier_a`, `tier_b`, `winner` or
`candidate_tiers`), so the notebook can choose any training objective —
per-prompt tier classification, winner-vs-loser contrast — without
rebuilding the data.

`router/pairs.py` (driven by `scripts/build_training_pairs.py`) derives the
training-ready form from that file, locally and with no network: one row per
**(prompt, tier, outcome, source)** sample for pairwise training — the model
learns whether a tier *wins, loses, or ties* on a prompt, then routes by scoring
tiers 1–5 and taking the argmax. Decisive battles emit (winner tier, 1.0) +
(loser tier, 0.0); ties/both-bad emit both tiers at 0.5; min-solving-tier rows
(RouterBench and ComplexityRouter) emit their solving tiers at 1.0 and every
other candidate at 0.0. Decisive same-tier battles are skipped (no complexity
gradient; `--keep-same-tier` to include). `--max-pairs N` caps the output with a
seeded, proportionally **stratified sample over (source, tier)** — never truncate
the file by head, rows are ordered by source era. `--min-tier-gap N` keeps only
battles whose tiers differ by ≥ N (adjacent-tier human votes are near coin-flips),
and `--sources` restricts which sources contribute.

```bash
python scripts/build_training_dataset.py                       # arena suite
python scripts/build_training_dataset.py --sources arena,routerbench,complexity_router
python scripts/build_training_dataset.py --limit 500           # smoke run
python scripts/build_training_pairs.py                         # all sources: 554,352 pairs
python scripts/build_training_pairs.py --min-tier-gap 2 --drop-ties  # the gap filter: 243,722
python scripts/build_training_pairs.py --sources complexity_router   # the live experiment: 20,000
python scripts/build_training_pairs.py --max-pairs 15000       # capped for quick training
```

Gates: per-source model coverage must be ≥ 99% (unmapped names go to
`overrides.yaml`, rebuild the registry, retry) and duplicate battles (same
prompt + model pair) across releases are dropped. Coverage is enforced for every
source that resolves model names — model-free sources are exempt because there
are no names, which the code derives rather than hardcodes. The build writes
`eval_reports/training_dataset_report.md` — per-source counts, label-tier
histograms, prompt-length stats; eyeball it after every build.

## Training

Training runs in **Colab (T4) only** — the local machine never trains.
`notebooks/train_router_colab.ipynb` is self-contained: upload
`data/router_training_pairs.jsonl` (built locally), and it fine-tunes
`microsoft/deberta-v3-base` (~86M params) as a **pairwise tier scorer** — the
model reads (prompt, tier description) and predicts win/lose/tie, using only the
`prompt`, `tier` and `outcome` fields (the `source` column is ignored). Routing =
argmax over the five fixed tier descriptions, which are saved alongside the
weights as `tier_texts.json` because they are part of the model contract.

Two details that came out of failed runs and should not be "simplified" away:
the loss is `BCEWithLogitsLoss(pos_weight=4.0)` — plain BCE has a zero-gradient
optimum at "always predict 0.2" that collapses into ignoring the tier text — and
DeBERTa needs `AdamW(eps=1e-6)` because the default 1e-8 NaNs. Non-finite-loss
batches are skipped rather than clipped. Manual PyTorch loop, no `pin_memory`,
DirectML loaded lazily — every accelerator path stays open. The validation split
is prompt-grouped (`sha1(prompt) % 20 == 0`) so a prompt's winning and losing
tiers never straddle it.

The metric that tracks progress is **pairwise accuracy** (gold tier scored above
loser tier), currently the number to watch climb toward 0.66–0.68. Routing
accuracy caps near the human label-agreement rate (~0.42) and must be read
against the majority-tier baseline the notebook prints beside it.

To re-check a checkpoint without training, `scripts/eval_checkpoint.py` runs
batched inference only (safe on the Mac) and reproduces the notebook's split and
numbers by importing them from `router/contracts.py` and `router/metrics.py`:

```bash
python scripts/eval_checkpoint.py --skip-train
python scripts/eval_checkpoint.py --checkpoint models/complexity_router/final --device cpu
```

The earlier MiniLM cross-encoder pipeline (capability/eval based) is retired; its
training script survives only in git history (`993610e`).

# AGENTS.md

## What this repo is

A homebrew LLM router, pivoted (Oct 2026) to **complexity-based routing**:
training data is labeled by translating every model name into a complexity
tier 1–5 (absolute capability: Opus-class = 5, flash/mini = 1). Step 1 — the
model registry — is done and committed. Step 2 — the (prompt → complexity)
training dataset builder — is done too: `scripts/build_training_dataset.py`
labels four LMArena battle releases + RouterBench through the resolver, plus
the model-free `RowRed/ComplexityRouter` annotations. Next step: train on it
in the Colab notebook (now fine-tuning `microsoft/deberta-v3-base`) and
decide per the standing protocol — pairwise acc climbing toward 0.66–0.68 →
scale up; flat → pivot the label source.

## Layout

- `router/model_registry.py` — runtime resolver (`get_registry().resolve(name)`)
  and the **single source of truth** for `normalize_name()` + `HAND_ALIASES`
- `scripts/build_model_registry.py` — merges 4 catalogs (LiteLLM, OpenRouter,
  models.dev, Epoch AI) and scores tiers → `data/model_registry/models.json`
- `scripts/audit_model_coverage.py` — resolves a dataset's model columns
  against the registry; coverage must stay ≥ 99% (`--all` audits the full
  training suite; every accessible source is at 100%; the model-free
  ComplexityRouter source has no names to audit and isn't listed)
- `scripts/build_training_dataset.py` — builds
  `data/router_training_dataset.jsonl` (gitignored): arena winner-tier
  labels + RouterBench min-solving-tier labels through the resolver, plus
  `complexity_router` (RowRed/ComplexityRouter: Qwen3.5-4B complexity
  levels 0–3 mapped linearly to tiers 1–4, `label_kind="annotated_level"`,
  `synthetic` flag marks the 813 synthetic rewrites; no resolver, so the
  coverage gate doesn't apply); hard-fails below 99% coverage per
  model-name source
- `scripts/build_training_pairs.py` — derives
  `data/router_training_pairs.jsonl` (gitignored) from that file locally:
  (prompt, tier, outcome 1.0/0.5/0.0, source) rows for pairwise training;
  skips same-tier decisive battles (no complexity gradient); routerbench-
  shaped rows (`candidate_tiers`/`solving_tiers`: routerbench AND
  complexity_router) emit solving tiers 1.0 / other candidates 0.0;
  `--max-pairs` applies a seeded stratified (source, tier) sample;
  `--sources` restricts which sources contribute — the on-disk file is
  currently complexity_router-only (20k pairs, complete 5-tier groups per
  prompt) for the active experiment; omit the flag for the merged file
  (~244k pairs under the gap filter)
- `data/model_registry/models.json` — **generated but committed**; never
  hand-edit. All hand tiers/aliases/created entries go in
  `data/model_registry/overrides.yaml`, then rebuild
- `data/model_registry/cache/`, `build_report.md`, `eval_reports/`,
  `models/` — gitignored
- `notebooks/train_router_colab.ipynb` — the only training environment
  (fine-tunes `microsoft/deberta-v3-base`; tokenizer needs `sentencepiece`)
- `training/` — empty; the old training script exists only in git history
  (`git show 993610e:training/train_router.py`). README sections below the
  registry section describe that retired capability/eval pipeline — don't
  trust them for current file names.

## Commands

```bash
.venv/bin/python scripts/build_model_registry.py [--refresh]   # regenerate models.json
.venv/bin/python scripts/audit_model_coverage.py --all         # audit the full training suite
.venv/bin/python scripts/build_training_dataset.py --sources arena,routerbench,complexity_router
.venv/bin/python scripts/build_training_pairs.py --sources complexity_router
.venv/bin/pyright router/ scripts/                             # must stay 0 errors
```

`lmsys/chatbot_arena_conversations` (33k) is gated: accept its license on
huggingface.co + `hf auth login`, then rerun the audit/build to include it.

Use the repo `.venv` (Python 3.14) — system python lacks the deps. No test
suite exists; validation = pyright clean + coverage audit + eyeballing the
tier sample table in `build_report.md`.

## Hard rules

- **Never start training runs locally** — the user's Mac cannot train;
  Colab T4 only (see notebook). Building datasets locally is fine.
- **Multi-accelerator is a hard requirement** (MPS/CUDA/ROCm/DirectML/CPU):
  never static-import `torch_directml` (use `importlib` via
  `device_utils._load_directml()`), avoid HF Trainer / DataLoader
  `pin_memory` — training uses a manual PyTorch loop for DirectML support.
- **pyright is pinned to `basic`** via `pyrightconfig.json` — do not delete
  or loosen it; keep 0 errors/0 warnings.
- **Registry layering**: scripts must import `normalize_name` from
  `router.model_registry`, never duplicate it. Alias resolution is
  HAND_ALIASES-first, so ambiguous marketing names ("gemini-pro") resolve to
  the historical original — intentional, training data is 2022–2025.
- Editing `.ipynb`: there is no NotebookEdit tool — patch the JSON with a
  Python script instead.

## Scoring invariants (don't break silently)

Tier scoring is era-aware; if you touch it, keep these:
- intelligence-index **downgrades only for models released ≥ 2025-01**
  (today's index is era-biased: gpt-4o's 2026 index is 8.4);
- design-arena ELO only **lifts** (≥ 1300 → frontier), never downgrades;
- legacy open models (pre-2024-09, known params) score by params, not
  hoster price; pre-2023-03 models are tier 2; era caps: <2024-09 → max 4.

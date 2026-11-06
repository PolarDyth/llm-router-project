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

`router/` is the domain: one module per pipeline stage, each importable and
testable. `scripts/` are **thin CLIs** — argument parsing, stage ordering, and
markdown report rendering, nothing else. If you find yourself adding a rule to a
script, it belongs in the stage module instead.

- `router/contracts.py` — the (prompt, tier) interface: `TIERS`, `TIER_TEXTS`,
  `MAX_LEN`, `VAL_PROMPT_MOD`, `COVERAGE_TARGET`, `COMPLEXITY_SCALE`. Single
  repo-side truth; the notebook's inline copy is tested against it
- `router/model_registry.py` — runtime resolver (`get_registry().resolve(name)`)
  and the **single source of truth** for `normalize_name()` + `HAND_ALIASES`
- `router/catalogs.py` — the four vendor catalogs (LiteLLM, OpenRouter,
  models.dev, Epoch AI): `CatalogCache` fetch + load, `merge_records` →
  `Canonical` groups. No tiering here on purpose
- `router/scoring.py` — canonical group → tier: price prior, quality adjust,
  era caps, `apply_overrides`, `alias_collisions`. The era/ELO/params
  invariants live here as named, individually tested functions
- `router/sources.py` — **one catalog of dataset sources**, shared by the audit
  and the build; `name_mode` decides whether a source carries model names and is
  therefore audited and coverage-gated. Also the RouterBench pkl column rules
- `router/dataset.py` — dataset-stage adapters (`arena_rows`,
  `routerbench_rows`, `complexity_router_rows`) as generators, arena schema
  normalization, and the `coverage_failures` gate
- `router/pairs.py` — dataset row → `(prompt, tier, outcome)` pairs; the single
  row-shape dispatch, the filtering rules, and `allocate`
- `router/metrics.py` — eval math: prompt-hash split, cohorts,
  pairwise/routing accuracy, majority baseline
- `router/devices.py` — cross-platform device selection (was root
  `device_utils.py`); imported by `scripts/eval_checkpoint.py`
- `scripts/build_model_registry.py` — CLI: acquire → merge → score → overrides →
  write `models.json` + report (`--cache/--out/--report` to build elsewhere)
- `scripts/audit_model_coverage.py` — CLI: resolves a dataset's model columns
  against the registry; coverage must stay ≥ 99% (`--all` audits
  `router.sources.AUDITABLE` — the same list the build gates, so the audit
  predicts the build; the model-free ComplexityRouter source has no names and
  isn't listed)
- `scripts/build_training_dataset.py` — CLI: builds
  `data/router_training_dataset.jsonl` (gitignored) from the requested source
  families; hard-fails below 99% coverage per source that resolved model names
  (exemption is derived from whether names were resolved, not from a
  hardcoded source list — this previously exempted RouterBench by mistake)
- `scripts/build_training_pairs.py` — CLI: derives
  `data/router_training_pairs.jsonl` (gitignored) locally in two streaming
  passes over the dataset; `--max-pairs` applies a seeded stratified
  (source, tier) sample; `--sources` restricts which sources contribute —
  the on-disk file is currently complexity_router-only (20k pairs, complete
  5-tier groups per prompt) for the active experiment; omit the flag for the
  merged file (~244k pairs under the gap filter)
- `scripts/eval_checkpoint.py` — local inference-only eval of a
  Colab-trained checkpoint
- `tests/` — pytest suite (299 tests, ~3 s): stage rules, the scoring
  invariants, the notebook↔repo contract, and `tests/test_no_duplication.py`
  which fails if a shared rule gets re-copied anywhere
- `tests/fixtures/dataset_sample.jsonl` — 600 real dataset rows (120 from each
  of the five redistributable sources, both row shapes, all three label kinds)
  and `tests/golden/pairs_manifest.json` — byte-fingerprints of 7 pairs builds
  captured from the **pre-restructure** code; both committed, both offline.
  `arena_33k` rows are excluded on purpose: that dataset is CC-BY-NC-4.0 and
  license-gated, and a test enforces that its rows never get checked in
- `data/model_registry/models.json` — **generated but committed**; never
  hand-edit. All hand tiers/aliases/created entries go in
  `data/model_registry/overrides.yaml`, then rebuild
- `data/model_registry/cache/`, `build_report.md`, `eval_reports/`,
  `models/`, `llm_router.egg-info/` — gitignored
- `notebooks/train_router_colab.ipynb` — the only training environment
  (fine-tunes `microsoft/deberta-v3-base`; tokenizer needs `sentencepiece`)
- `training/` — gone; the old training script exists only in git history
  (`git show 993610e:training/train_router.py`)

## Commands

```bash
.venv/bin/pip install -e ".[dev]" --no-deps   # once: makes `router` importable
.venv/bin/python -m pytest -q                 # the whole suite, ~3 s, offline
.venv/bin/pyright router/ scripts/ tests/     # must stay 0 errors / 0 warnings
.venv/bin/python scripts/build_model_registry.py [--refresh]   # regenerate models.json
.venv/bin/python scripts/audit_model_coverage.py --all         # audit the full training suite
.venv/bin/python scripts/build_training_dataset.py --sources arena,routerbench,complexity_router
.venv/bin/python scripts/build_training_pairs.py --sources complexity_router
.venv/bin/python scripts/eval_checkpoint.py --skip-train       # local inference-only eval
```

The scripts import `router` as an installed package — there is no `sys.path`
patching anywhere, and `pip install -e .` is required before anything runs.
`--no-deps` keeps pip from re-resolving the carefully pinned torch build.

`lmsys/chatbot_arena_conversations` (33k) is gated: accept its license on
huggingface.co + `hf auth login`, then rerun the audit/build to include it.

Use the repo `.venv` (Python 3.14) — system python lacks the deps. Validation is
`pytest` + pyright clean + the coverage audit + eyeballing the tier sample table
in `build_report.md`. Tests needing the vendor catalog cache skip themselves on a
fresh clone (`build_model_registry.py --refresh` populates it); everything else
runs offline against the committed fixture.

## Hard rules

- **Never start training runs locally** — the user's Mac cannot train;
  Colab T4 only (see notebook). Building datasets locally is fine.
- **Multi-accelerator is a hard requirement** (MPS/CUDA/ROCm/DirectML/CPU):
  never static-import `torch_directml` (use `importlib` via
  `router/devices._load_directml()`), avoid HF Trainer / DataLoader
  `pin_memory` — training uses a manual PyTorch loop for DirectML support.
- **pyright is pinned to `basic`** via `pyrightconfig.json` — do not delete
  or loosen it; keep 0 errors/0 warnings across `router/`, `scripts/` **and
  `tests/`**.
- **Rules live in `router/`, not in `scripts/`.** A script parses args, calls
  stage modules, and renders a report. `tests/test_no_duplication.py` asserts
  single definitions for every shared rule (`normalize_name`, the row-shape
  dispatch, device selection, the pkl column rule, the contract constants) and
  that nothing patches `sys.path` — it will fail if a copy creeps back in.
- **Never commit rows from a gated license.** `lmsys/chatbot_arena_conversations`
  (CC-BY-NC-4.0) and anything else requiring a Hub license accept may be read
  locally to build the gitignored `data/*.jsonl`, but its rows must not land in
  `tests/fixtures/`, docs, or any tracked file — this repo has a GitHub remote.
  `tests/test_registry_rebuild.py::test_fixture_contains_no_rows_from_gated_sources`
  enforces it.
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

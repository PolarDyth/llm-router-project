# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` is the live project doc (current pivot status, layout map, commands,
hard rules, scoring invariants) and is loaded into context automatically — read
it first. This file adds the parts that only show up when you read several
files at once: the cross-file contracts between the pipeline stages, and which
docs are stale.

## Pipeline: five stages, each stage's output is the next stage's input

Each stage's *logic* lives in a `router/` module; the script is only its CLI.

```
4 vendor catalogs → catalogs.py + scoring.py        → models.json   (tier per canonical model)
models.json     → sources.py + dataset.py           → router_training_dataset.jsonl  (model names → tiers)
dataset.jsonl   → pairs.py                          → router_training_pairs.jsonl    (prompt, tier, outcome)
pairs.jsonl     → notebooks/train_router_colab.ipynb → models/complexity_router/final/
checkpoint      → contracts.py + metrics.py         → scripts/eval_checkpoint.py     (local inference-only eval)
```

The remaining files are cross-cutting: `model_registry.py` (the resolver every
labeled row passes through), `devices.py` (accelerator selection), and
`contracts.py` (the (prompt, tier) constants shared with the notebook).

Every stage is regenerable from the one above it; only `models.json` +
`overrides.yaml` are committed. That makes the **invalidation cascade** the
thing to know before changing anything:

- Edit `overrides.yaml` / `HAND_ALIASES` / `normalize_name()` → the tier of
  already-labeled rows silently changes → rebuild registry, then
  `audit_model_coverage.py --all`, then rebuild the dataset, then rebuild pairs.
  Skipping the dataset rebuild leaves labels computed under old normalization.
- `build_training_dataset.py` and `audit_model_coverage.py` both call
  `get_registry()`, which is `@lru_cache(maxsize=1)` on
  `data/model_registry/models.json` — a fresh process per build is what picks up
  a rebuilt registry, not a reload.
- `data/router_training_dataset.jsonl` is ~365 MB; the pairs step is a local
  two-pass streaming read (`strata_counts` then the emit pass), so keep any new
  filter in that shape rather than materializing rows.

## Contract 1: two row shapes in the dataset, dispatched by field presence

`router/pairs.pairs_from_row()` does not switch on `source` — it branches on the
row. This is now the only dispatch site; before the restructure the pairs build
inlined it twice (once per streaming pass), so the strata count and the emitted
subset could disagree:

- `candidate_tiers` is a list → `solving_tier_pairs()` (routerbench **and**
  complexity_router): solving tiers at 1.0, other candidates at 0.0.
- otherwise → `arena_pairs()` using `tier_a`/`tier_b`/`winner`.

So a new dataset source must either emit one of those two shapes (complexity_router
cheats by reusing the routerbench shape with `solving_tiers=[label_tier]`) or add a
branch here. `label_kind` (`winner_tier` / `min_solving_tier` / `annotated_level`)
is provenance for humans and the notebook, not a dispatch key.

The 99% coverage gate (`router/contracts.COVERAGE_TARGET`, enforced by
`router/dataset.coverage_failures`) is applied to any source that resolved
model-name slots, and hard-fails the build before a file is written. Exemption
is *derived* from `SourceStats.name_slots > 0` rather than from a list of source
keys, because that list was wrong: it declared routerbench "model-free" and
returned 0 slots for it, so the source carrying ~401k model names was the one
source the build never gated (the audit did cover it, at 100%). A new source that
names models is now gated automatically, with nothing to remember to update.

## Contract 2: the (prompt, tier) interface, with one forced duplicate

Training and inference must tokenize identically. The repo-side truth is
`router/contracts.py` (`TIERS`, `TIER_TEXTS`, `MAX_LEN`, `VAL_PROMPT_MOD`,
`COVERAGE_TARGET`, `COMPLEXITY_SCALE`). Exactly one copy still exists outside
it — the Colab notebook, which runs with only the pairs file uploaded and has no
checkout to import:

| constant | `router/contracts.py` | notebook cells 9/11/13 | `scripts/eval_checkpoint.py` |
|---|---|---|---|
| `TIER_TEXTS` (5 strings) | source of truth | re-declares (unavoidable) | read from the checkpoint's `tier_texts.json` |
| `MAX_LEN = 256` | source of truth | re-declares | **imports it** |
| `VAL_PROMPT_MOD = 20` | source of truth | `sha1(prompt) % 20 == 0` → val | **imports it** |
| `num_labels=1` scalar head | — | `AutoModelForSequenceClassification` | must pass `num_labels=1` again (it is not saved in `config.json`) |

`tests/test_contracts.py` parses the notebook JSON and `literal_eval`s its
assignments, so notebook drift fails the suite instead of silently leaking
held-out prompts into "train". That guard is what replaced the old
"remember to update both" discipline. Routing is argmax over the five tier
scores; the tier strings are part of the model contract, so editing `TIER_TEXTS`
invalidates every existing checkpoint — the checkpoint's own `tier_texts.json`
is asserted equal to the contract. Routing is always argmax over
the five tier scores; the tier strings are part of the model contract, so
editing `TIER_TEXTS` invalidates every existing checkpoint.

## Contract 3: why the loss and the metrics are shaped like they are

The pairs file gives every prompt exactly one 1.0 gold and four 0.0 losers,
which is load-bearing in three places:

- the notebook trains with `BCEWithLogitsLoss(pos_weight=4.0)`, *not* plain BCE:
  unweighted BCE has a zero-gradient optimum at "always predict 0.2", which
  collapses to ignoring the tier text (run 2: identical scores across tiers,
  pairwise acc 0.000). `pos_weight` matches the 4:1 balance — revisit it if the
  data mix changes, e.g. when arena tie rows (0.5 targets) come back in.
- `routing_accuracy` only counts prompts with exactly one 1.0 and is capped near
  the human label-agreement rate (~0.42); `pairwise_accuracy` (winner tier
  scored above loser tier) is the metric that actually tracks progress — that's
  the one climbing toward 0.66–0.68 in `AGENTS.md`.
- `router/metrics.py` owns the "exactly one gold per prompt" cohort filter
  (`metric_cohort`, `gold_tiers`), the prompt-hash split, and all three numbers;
  `scripts/eval_checkpoint.py` imports them. The notebook re-declares the same
  math inline (no Colab checkout), and `tests/test_metrics.py` pins the
  definitions so the three stay comparable rather than merely similar.

Non-finite-loss batches are skipped rather than clipped (an Inf that survives
clipping poisons all weights), and DeBERTa needs `AdamW(eps=1e-6)` — the default
1e-8 is NaN-prone. Those two are in the notebook; keep them if you refactor it.

## Device handling

`router/devices.select_device()` (cuda → mps → directml → cpu) is the repo-local
entry point and `scripts/eval_checkpoint.py` imports it. The only surviving
inline copy is the notebook's `pick_device()`, which is forced by Colab having no
repo checkout — that duplication is deliberate, not a refactor target. (The
root-level `device_utils.py` this section used to describe had *zero* importers
after the training script was retired; `eval_checkpoint.py` carried a divergent
private copy instead. Both are gone.)

`select_device` validates `--device` against `VALID_DEVICES` and exits with an
install hint when a forced device is unavailable, where the old inline copy built
`torch.device(anything)` and let it fail later. DirectML is always reached through
`importlib` (never a static import) so it stays optional on platforms where it
can't be installed, and callers pass the returned `torch.device` around as-is
since the DirectML one is a `privateuseone` object that can't be rebuilt from a
string. No `pin_memory` anywhere; the training loop is manual (no HF Trainer).
`tests/test_no_duplication.py` asserts all of that stays true.

## Which docs to trust

- `AGENTS.md` — current. Source of truth for rules and the layout map.
- `README.md` — current, including the Training section (the retired MiniLM
  cross-encoder description was replaced with the DeBERTa pairwise scorer and
  the stale pair counts corrected).
- Code + the notebook markdown cells — most current of all; the notebook's
  comments record failed runs (1e-5 LR collapse, NaN history) and why each
  constant is what it is.
- When the three disagree, `tests/` settles it: the suite encodes the scoring
  invariants, the notebook contract, and the pre-restructure output
  fingerprints, so a doc claim that contradicts a passing test is the stale one.

## Working state to not be surprised by

`data/router_training_pairs.jsonl` on disk is deliberately source-restricted —
20k rows, `complexity_router` only, exactly 4000 per tier, complete 5-tier
groups per prompt — for the active single-source experiment. It is not a broken
or truncated build; re-run `build_training_pairs.py --sources complexity_router`
to reproduce it, or omit `--sources` for the merged file. The old `training/`
directory and `scripts/test_router.py` are gone — `training/` survived only as
orphaned bytecode of a deleted source, now removed; both live in git history
(`git show 993610e:training/train_router.py`).

There is a pytest suite now (`tests/`, 299 tests, ~3 s, no network) plus pyright;
there is still no linter. Validation is `.venv/bin/python -m pytest -q` and
`.venv/bin/pyright router/ scripts/ tests/` (both must stay clean, `basic` mode
per `pyrightconfig.json`), alongside the coverage audit, a `--limit 500` smoke
build, and eyeballing `eval_reports/training_dataset_report.md` plus the tier
histogram in `data/model_registry/build_report.md`.

What the suite is for, beyond arithmetic: `test_scoring.py` pins the era/ELO/
params invariants AGENTS.md says not to break silently; `test_contracts.py`
parses the notebook and fails if its constants drift; `test_pairs_cli.py`
byte-compares seven pairs builds against fingerprints captured before the
restructure; `test_registry_rebuild.py` rebuilds `models.json` from the cached
catalogs and requires it to match what is committed. Quick ways to check one
stage without touching the rest:

```bash
.venv/bin/python -m pytest -q                                     # everything
.venv/bin/python -m pytest tests/test_scoring.py -q               # just tier logic
.venv/bin/python scripts/build_model_registry.py --refresh        # re-download catalogs
.venv/bin/python scripts/build_model_registry.py --out /tmp/m.json --report /tmp/r.md  # build elsewhere
.venv/bin/python scripts/build_training_dataset.py --limit 500 --sources complexity_router
.venv/bin/python -c "from router.model_registry import get_registry as g; print(g().complexity('gpt-4-1106-preview'))"   # 4
.venv/bin/python scripts/audit_model_coverage.py --file data/foo.csv --columns model,model_id
```

`pip install -e .` is a prerequisite for all of it — the scripts import `router`
as a package and nothing patches `sys.path` anymore.

Notebook edits: there is no NotebookEdit tool here, so patch the `.ipynb` JSON
with a Python script (see `AGENTS.md`), and keep the markdown cells' prose in
sync with the constants they describe.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` is the live project doc (current pivot status, layout map, commands,
hard rules, scoring invariants) and is loaded into context automatically — read
it first. This file adds the parts that only show up when you read several
files at once: the cross-file contracts between the pipeline stages, and which
docs are stale.

## Pipeline: five stages, each stage's output is the next stage's input

```
4 vendor catalogs → build_model_registry.py → models.json   (tier per canonical model)
models.json     → build_training_dataset.py  → router_training_dataset.jsonl  (model names → tiers)
dataset.jsonl   → build_training_pairs.py    → router_training_pairs.jsonl    (prompt, tier, outcome)
pairs.jsonl     → notebooks/train_router_colab.ipynb → models/complexity_router/final/
checkpoint      → scripts/eval_checkpoint.py         (local inference-only eval)
```

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

`build_training_pairs.py` does not switch on `source` — it branches on the row:

- `candidate_tiers` is a list → `solving_tier_pairs()` (routerbench **and**
  complexity_router): solving tiers at 1.0, other candidates at 0.0.
- otherwise → `arena_pairs()` using `tier_a`/`tier_b`/`winner`.

So a new dataset source must either emit one of those two shapes (complexity_router
cheats by reusing the routerbench shape with `solving_tiers=[label_tier]`) or add a
branch here. `label_kind` (`winner_tier` / `min_solving_tier` / `annotated_level`)
is provenance for humans and the notebook, not a dispatch key.

The 99% coverage gate in `build_training_dataset.py` (`COVERAGE_TARGET`) is
applied per model-name source and hard-fails the build; model-free sources are
exempt because there are no names to resolve — anything added later that does
carry model names needs to be added to that gate's list.

## Contract 2: the (prompt, tier) interface, duplicated on purpose

Training and inference must tokenize identically. These constants exist in two
places that cannot import each other (the notebook runs in Colab with only the
pairs file uploaded; `device_utils.py` isn't there either):

| constant | notebook cells 9/11/13 | `scripts/eval_checkpoint.py` |
|---|---|---|
| `TIER_TEXTS` (5 strings) | source of truth | read from the checkpoint's `tier_texts.json` |
| `MAX_LEN = 256` | yes | yes — hardcoded |
| `VAL_PROMPT_MOD = 20` | `sha1(prompt) % 20 == 0` → val | same expression |
| `num_labels=1` scalar head | `AutoModelForSequenceClassification` | must pass `num_labels=1` again |

Change `MAX_LEN` or the split modulus in the notebook → update
`eval_checkpoint.py` too, or the local numbers stop being the notebook's
numbers and held-out prompts leak into "train". Routing is always argmax over
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
- `eval_checkpoint.py`'s majority baseline and `metric_cohort()` re-implement
  the same "exactly one gold per prompt" filter to be comparable to the
  notebook.

Non-finite-loss batches are skipped rather than clipped (an Inf that survives
clipping poisons all weights), and DeBERTa needs `AdamW(eps=1e-6)` — the default
1e-8 is NaN-prone. Those two are in the notebook; keep them if you refactor it.

## Device handling

`device_utils.select_device()` (cuda → mps → directml → cpu) is for repo-local
scripts. The notebook and `eval_checkpoint.py` each carry an inline
`pick_device()` copy because Colab has no repo checkout — the duplication is
deliberate, not a refactor target. DirectML is always reached through
`importlib` (never a static import) so it stays optional on platforms where it
can't be installed, and callers pass the returned `torch.device` around as-is
since the DirectML one is a `privateuseone` object that can't be rebuilt from a
string. No `pin_memory` anywhere; the training loop is manual (no HF Trainer).

## Which docs to trust

- `AGENTS.md` — current. Source of truth for rules and the layout map.
- `README.md` — accurate through the registry/training-data sections, but its
  **Training** section still describes the retired MiniLM cross-encoder pipeline
  and its pair counts are stale (~534k; ~244k merged under the gap filter).
  Everything under "Training" in README is pre-pivot.
- Code + the notebook markdown cells — most current of all; the notebook's
  comments record failed runs (1e-5 LR collapse, NaN history) and why each
  constant is what it is.

## Working state to not be surprised by

`data/router_training_pairs.jsonl` on disk is deliberately source-restricted —
20k rows, `complexity_router` only, exactly 4000 per tier, complete 5-tier
groups per prompt — for the active single-source experiment. It is not a broken
or truncated build; re-run `build_training_pairs.py` without `--sources` for the
merged file. Likewise, `training/` is empty and `scripts/test_router.py` is gone:
both live only in git history (`git show 993610e:training/train_router.py`).

There is no test suite and no linter beyond pyright, so validation is:
`.venv/bin/pyright router/ scripts/` (must stay 0 errors/0 warnings,
`basic` mode per `pyrightconfig.json`), the coverage audit, a `--limit 500`
smoke build, and eyeballing `eval_reports/training_dataset_report.md` plus the
tier histogram in `data/model_registry/build_report.md`. Quick ways to check
one stage without touching the rest:

```bash
.venv/bin/python scripts/build_model_registry.py --refresh        # re-download catalogs
.venv/bin/python scripts/build_training_dataset.py --limit 500 --sources complexity_router
.venv/bin/python -c "from router.model_registry import get_registry as g; print(g().complexity('gpt-4-1106-preview'))"   # 4
.venv/bin/python scripts/audit_model_coverage.py --file data/foo.csv --columns model,model_id
```

Notebook edits: there is no NotebookEdit tool here, so patch the `.ipynb` JSON
with a Python script (see `AGENTS.md`), and keep the markdown cells' prose in
sync with the constants they describe.

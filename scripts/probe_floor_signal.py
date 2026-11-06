#!/usr/bin/env python3
"""Pre-training gate: is a capability-floor label actually learnable, and does it
transfer to a later era, before anyone spends a Colab session finding out?

Why this exists. The label rules are argued, the corpus is measured, but whether a
model can learn `prompt -> tier` from ~7k mostly-single-vote labels, and whether
what it learns generalizes past 2023-24 phrasing, are empirical questions. A bag
of words plus logistic regression answers the cheap version of both in seconds,
offline, on a laptop -- no GPU, no training run (which stays Colab-only).

It is a FLOOR, not a forecast: DeBERTa-v3 has 86M pretrained parameters and will
clear a TF-IDF baseline. What the probe measures is direction:

  same-era CV accuracy   is there learnable signal in this label at all
  out-of-era accuracy    does that signal survive a change of era -- the number
                         that predicts real routing traffic, since the router is
                         deployed on 2026 prompts and trained on 2023-25 ones
  era predictability     how much of the classifier's apparent skill could be
                         "which year is this prompt from" instead of difficulty

Prompts are truncated to MAX_LEN tokens here, exactly as training will truncate
them, so the probe sees the same information the model will get.

Usage:
  python scripts/probe_floor_signal.py
  python scripts/probe_floor_signal.py --floor data/router_training_floor.jsonl \
      --holdout data/router_floor_holdout.jsonl --report eval_reports/floor_probe.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from router.contracts import MAX_LEN, TIERS

ROOT = Path(__file__).resolve().parent.parent

FLOOR_DEFAULT = ROOT / "data" / "router_training_floor.jsonl"
HOLDOUT_DEFAULT = ROOT / "data" / "router_floor_holdout.jsonl"
REPORT_DEFAULT = ROOT / "eval_reports" / "floor_probe.md"
TOKENIZER_DEFAULT = ROOT / "models" / "complexity_router" / "final"


def load_labeled(path: Path, max_len: int, tokenizer: Any | None) -> list[dict[str, Any]]:
    """Floor rows reduced to (truncated prompt, gold tier, provenance)."""
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            tier = r.get("label_tier")
            prompt = str(r.get("prompt", ""))
            if tier is None or not prompt.strip():
                continue
            if tokenizer is not None:
                ids = tokenizer(prompt, add_special_tokens=False, max_length=max_len,
                                truncation=True)["input_ids"]
                text = tokenizer.decode(ids)
            else:  # stated fallback: same 4-chars-per-token estimate the selector uses
                text = prompt[: max_len * 4]
            rows.append({"text": text, "tier": int(tier),
                         "n_constraints": int(r.get("n_constraints", 1)),
                         # newest release that evidences this prompt -- the era
                         # label the shortcut measurement predicts from text
                         "eras": tuple(sorted(r.get("supporting_sources") or {}))})
    return rows


def vectorize(texts: list[str]) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60_000,
                           sublinear_tf=True).fit_transform(texts)


def classifier() -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=2000, C=4.0)


def per_tier_recall(y: Any, pred: Any) -> dict[int, float]:
    return {t: round(float((pred[y == t] == t).mean()), 3)
            for t in sorted(set(y.tolist())) if (y == t).sum() > 0}


def probe(floor: list[dict[str, Any]], holdout: list[dict[str, Any]],
          balanced_prior: float) -> dict[str, Any]:
    """The three measurements. sklearn is imported lazily so --help stays cheap."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import cross_val_predict

    out: dict[str, Any] = {}
    X = [r["text"] for r in floor]
    y = np.array([r["tier"] for r in floor])
    V = vectorize(X)

    pred = cross_val_predict(classifier(), V, y, cv=5)
    out["same_era"] = {"n": len(y), "accuracy": round(accuracy_score(y, pred), 3),
                       "macro_f1": round(f1_score(y, pred, average="macro"), 3),
                       "per_tier_recall": per_tier_recall(y, pred)}

    # Corroboration split: if single-vote labels are noisier, the model should do
    # visibly worse on them -- which is how R4 (91.5% one battle per prompt) gets
    # attributed instead of hand-waved.
    pred_arr, y_arr = np.asarray(pred), np.asarray(y)

    def recall_on(mask: Any) -> float:
        sel = np.asarray(mask, dtype=bool)
        return float(np.count_nonzero(pred_arr[sel] == y_arr[sel])) / int(sel.sum())

    k2 = np.array([r["n_constraints"] >= 2 for r in floor])
    if int(k2.sum()) >= 20 and int((~k2).sum()) >= 20:
        out["cv_recall_corroborated"] = round(recall_on(k2), 3)
        out["cv_recall_single_vote"] = round(recall_on(~k2), 3)
        out["n_corroborated"] = int(k2.sum())

    if holdout:
        Xh = [r["text"] for r in holdout]
        yh = np.array([r["tier"] for r in holdout])
        # The vectorizer is fit on TRAIN text only and reused to transform the
        # holdout: fitting it on both would let holdout vocabulary choice leak
        # into the features and inflate the one number that matters most here.
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60_000,
                              sublinear_tf=True)
        Vt = vec.fit_transform(X)
        model = classifier().fit(Vt, y)
        P = model.predict(vec.transform(Xh))
        # The comparison cohort's own prior, not the core's. The holdout covers
        # only the tiers that release evidenced, so it is usually far more
        # balanced than the training set -- and a predictor that beats the
        # *training* prior can still sit below the *holdout* prior, which is the
        # number that says whether anything transferred. Reporting one prior
        # against the other is how a below-chance result reads as a pass.
        hold_prior = float(max(Counter(yh.tolist()).values())) / len(yh)
        out["out_of_era"] = {
            "n": len(yh),
            "tier_mix": {int(t): int((yh == t).sum()) for t in sorted(set(yh.tolist()))},
            "majority_prior": round(hold_prior, 3),
            "accuracy": round(float(accuracy_score(yh, P)), 3),
            "macro_f1": round(float(f1_score(yh, P, average="macro")), 3),
            "per_tier_recall": {int(k): v for k, v in per_tier_recall(yh, P).items()},
        }

    # How predictable is the era itself? High here + low out-of-era = the
    # classifier is reading the year, not the difficulty.
    era_labels = np.array([r["eras"][-1] if r["eras"] else "?" for r in floor])
    if len(set(era_labels.tolist())) > 1:
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(classifier(), V, era_labels, cv=3)
        base = float(Counter(era_labels.tolist()).most_common(1)[0][1]) / len(era_labels)
        out["era_predictability"] = {"accuracy": round(float(scores.mean()), 3),
                                     "majority_base": round(base, 3)}
    out["balanced_prior"] = round(balanced_prior, 3)
    return out


def render(out: dict[str, Any], args: argparse.Namespace, path: Path) -> None:
    se = out["same_era"]
    lines = ["# floor signal probe", "",
             f"floor: `{args.floor}` | holdout: `{args.holdout}` | "
             f"truncation: {args.max_len} tokens (the training contract)",
             "backend: TF-IDF (1-2 grams) + logistic regression -- a floor on what "
             "DeBERTa can learn, not a forecast of it.", ""]
    lines.append(f"## same-era 5-fold CV: accuracy {se['accuracy']:.3f}, "
                 f"macro-F1 {se['macro_f1']:.3f} (balanced prior "
                 f"{out['balanced_prior']:.3f}, n={se['n']})")
    lines.append("")
    lines.append("| tier | CV recall |")
    lines.append("|---|---|")
    for t, v in sorted(se["per_tier_recall"].items()):
        lines.append(f"| {t} | {v:.3f} |")
    lines.append("")
    if "cv_recall_corroborated" in out:
        lines.append(f"Single-vote vs corroborated labels (n={out['n_corroborated']} "
                     f"corroborated): recall {out['cv_recall_single_vote']:.3f} for "
                     f"`n_constraints == 1` vs {out['cv_recall_corroborated']:.3f} for "
                     f"`>= 2`. A wide gap says most of the core's labels are the noisy "
                     "kind, and the headline number belongs to the minority.")
        lines.append("")
    if "out_of_era" in out:
        oo = out["out_of_era"]
        prior = oo.get("majority_prior", out["balanced_prior"])
        if oo["accuracy"] > prior:
            verdict = "BEATS the holdout's own majority prior"
        elif oo["accuracy"] > 1.0 / max(len(oo["tier_mix"]), 1):
            verdict = ("BEATS uniform chance but not the holdout's majority prior -- "
                       "a constant guesser would score higher than this model")
        else:
            verdict = ("is at or below uniform chance -- nothing transferred")
        lines.append(f"## out-of-era transfer: accuracy {oo['accuracy']:.3f}, "
                     f"macro-F1 {oo['macro_f1']:.3f} (n={oo['n']}) vs majority prior "
                     f"{prior:.3f} -- {verdict}")
        lines.append("")
        lines.append(f"Holdout tier mix: {oo['tier_mix']}. Tiers absent from the "
                     "holdout release cannot be evaluated out-of-era at all, so this "
                     "number describes the top of the scale only -- averaging it over "
                     "five tiers would misreport it.")
        lines.append("")
        lines.append("| tier | out-of-era recall |")
        lines.append("|---|---|")
        for t, v in sorted(oo["per_tier_recall"].items()):
            lines.append(f"| {t} | {v:.3f} |")
        lines.append("")
    if "era_predictability" in out:
        ep = out["era_predictability"]
        lines.append(f"## era shortcut: release of evidence is predictable from prompt "
                     f"text alone at {ep['accuracy']:.3f} (majority base "
                     f"{ep['majority_base']:.3f}).")
        lines.append("")
        lines.append("That is the confound the era-balanced selection trades against. "
                     "Read together with the out-of-era number: high here and low there "
                     "means a classifier that looks competent in-era is largely "
                     "identifying which year a prompt was written in.")
    lines.append("")
    lines.append("## what to do with this")
    lines.append("")
    lines.append("- This probe is a floor. DeBERTa should beat it; a model that beats "
                 "the same-era number but not the out-of-era one has learned era "
                 "vocabulary, not complexity.")
    lines.append("- Report the out-of-era number as the headline for any experiment "
                 "trained on this core, and the same-era number only alongside it.")
    lines.append("- If out-of-era sits at or below the balanced prior, the honest "
                 "conclusion is that these labels support coarse easy/mid/hard "
                 "ordering, not calibrated absolute 1-5.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=Path, default=FLOOR_DEFAULT)
    ap.add_argument("--holdout", type=Path, default=HOLDOUT_DEFAULT)
    ap.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    ap.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DEFAULT)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    args = ap.parse_args()

    if not args.floor.exists():
        raise SystemExit(f"missing {args.floor} - run "
                         f"scripts/select_floor_dataset.py first")

    tokenizer = None
    if (args.tokenizer_dir / "tokenizer.json").exists():
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
        except Exception as exc:                            # pragma: no cover
            print(f"[warn] no tokenizer ({exc}); truncating by chars/4")

    floor = load_labeled(args.floor, args.max_len, tokenizer)
    holdout = (load_labeled(args.holdout, args.max_len, tokenizer)
               if args.holdout.exists() else [])
    hist = Counter(r["tier"] for r in floor)
    if not hist:
        raise SystemExit(f"{args.floor} has no labeled prompts")
    balanced_prior = max(hist.values()) / sum(hist.values())
    print(f"floor prompts: {len(floor)} | tier mix: "
          f"{dict(sorted(hist.items()))} | holdout: {len(holdout)}")

    out = probe(floor, holdout, balanced_prior)
    se = out["same_era"]
    print(f"same-era CV:      acc {se['accuracy']:.3f}  macro-F1 "
          f"{se['macro_f1']:.3f}  (balanced prior {out['balanced_prior']:.3f})")
    if "out_of_era" in out:
        oo = out["out_of_era"]
        print(f"out-of-era:       acc {oo['accuracy']:.3f}  macro-F1 "
              f"{oo['macro_f1']:.3f}  n={oo['n']}  tiers={oo['tier_mix']}  "
              f"(holdout majority prior {oo.get('majority_prior', '?')})")
    if "era_predictability" in out:
        ep = out["era_predictability"]
        print(f"era shortcut:     predict release from text acc {ep['accuracy']:.3f} "
              f"(base {ep['majority_base']:.3f})")
    render(out, args, args.report)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()

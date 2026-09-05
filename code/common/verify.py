"""Official-evaluator validation: run the exact same aux-metric computation as
`evaluation.evaluate_submission.get_metric` on a submission file and assert
validity/uniqueness/novelty are above the grading threshold.

We mirror the eval code line-for-line so that any disagreement with our internal
post-processing surfaces here instead of after a submission is burned.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .paths import CHALLENGE_ROOT, EVAL_TRAIN_FILE
from .postprocess import canonicalize

THRESHOLD = 0.9


def official_aux_metrics(
    submission_path: Path, train_path: Path = EVAL_TRAIN_FILE
) -> dict:
    """Reproduce `evaluation.evaluate_submission.get_metric` aux-metric branches.

    Returns a dict with validity, uniqueness, novelty, plus the raw counts so
    callers can diagnose any shortfall.
    """
    with open(train_path) as f:
        smiles_train = {s for s in f.read().split() if s}

    with open(submission_path) as f:
        smiles_gen = [s for s in f.read().split() if s][:10_000]

    smiles_can = canonicalize(smiles_gen)
    smiles_valid = [s for s in smiles_can if s is not None]
    smiles_unique = set(smiles_valid)
    smiles_novel = smiles_unique - smiles_train

    denom = len(smiles_gen)  # the eval divides by this, NOT by 10_000 if file is short
    return {
        "n_in_file": denom,
        "n_valid": len(smiles_valid),
        "n_unique": len(smiles_unique),
        "n_novel": len(smiles_novel),
        "validity": len(smiles_valid) / denom,
        "uniqueness": len(smiles_unique) / denom,
        "novelty": len(smiles_novel) / denom,
    }


def assert_thresholds(metrics: dict, threshold: float = THRESHOLD) -> None:
    """Raise ValueError if any of validity/uniqueness/novelty is below threshold."""
    failed = [
        k for k in ("validity", "uniqueness", "novelty") if metrics[k] < threshold
    ]
    if failed:
        raise ValueError(
            f"Aux metrics below {threshold}: " + ", ".join(
                f"{k}={metrics[k]:.4f}" for k in failed
            ) + f"   full metrics: {metrics}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m common.verify <submission_path>", file=sys.stderr)
        sys.exit(2)
    p = Path(sys.argv[1])
    m = official_aux_metrics(p)
    for k, v in m.items():
        print(f"  {k:>11} = {v}")
    assert_thresholds(m)
    print(f"OK — all aux metrics >= {THRESHOLD}")

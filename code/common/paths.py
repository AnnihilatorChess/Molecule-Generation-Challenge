"""Shared paths for the Generation Challenge code."""
from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CHALLENGE_ROOT = REPO_ROOT / "Generation-Challenge"

DATA_DIR = CHALLENGE_ROOT / "data"
TRAIN_FILE = DATA_DIR / "smiles_train.txt"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.txt"
EVAL_DIR = DATA_DIR / "evaluation"
EVAL_TRAIN_FILE = EVAL_DIR / "data" / "smiles_train.txt"
EVAL_TEST_STATS = EVAL_DIR / "data" / "test_stats.p"

CODE_DIR = CHALLENGE_ROOT / "code"
ARTIFACTS_DIR = CHALLENGE_ROOT / "artifacts"
PREDICTIONS_DIR = CHALLENGE_ROOT / "predictions"


def version_dirs(version: str) -> tuple[Path, Path, Path]:
    """Return (code, artifacts, predictions) dirs for the given version."""
    code = CODE_DIR / version
    art = ARTIFACTS_DIR / version
    preds = PREDICTIONS_DIR / version
    art.mkdir(parents=True, exist_ok=True)
    preds.mkdir(parents=True, exist_ok=True)
    return code, art, preds

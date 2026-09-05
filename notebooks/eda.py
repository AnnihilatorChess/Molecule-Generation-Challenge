"""Exploratory data analysis for the Generation Challenge.

Reads the training SMILES, computes summary statistics on a random subsample
(properties + scaffolds are expensive on 1.27M molecules), and dumps a JSON
report + a few PNGs into notebooks/eda_out/.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

random.seed(1234)
np.random.seed(1234)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAIN = ROOT / "data" / "smiles_train.txt"
EVAL_TRAIN = ROOT / "data" / "evaluation" / "data" / "smiles_train.txt"
OUT = HERE / "eda_out"
OUT.mkdir(exist_ok=True)


def load_smiles(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    print(f"Reading {TRAIN} ...")
    train = load_smiles(TRAIN)
    print(f"  user-facing training set:  {len(train):,} SMILES")
    print(f"Reading {EVAL_TRAIN} ...")
    eval_train = load_smiles(EVAL_TRAIN)
    print(f"  evaluation-bundled novelty reference: {len(eval_train):,} SMILES")

    # Quick identity check between the two training files
    same_set = (set(train) == set(eval_train))
    print(f"  user_train set == eval_train set: {same_set}")

    # ---- Length statistics on full corpus ----
    lengths = np.array([len(s) for s in train], dtype=np.int32)
    print(
        f"\nSMILES length: min={lengths.min()} max={lengths.max()} "
        f"mean={lengths.mean():.1f} median={int(np.median(lengths))} "
        f"p95={int(np.percentile(lengths, 95))} p99={int(np.percentile(lengths, 99))}"
    )

    # ---- Character vocabulary on full corpus ----
    char_counter: Counter = Counter()
    for s in train:
        char_counter.update(s)
    print(f"\nDistinct characters: {len(char_counter)}")
    print("Top 30 chars:", char_counter.most_common(30))

    # ---- Multi-char atom tokens (Cl, Br, [...]) ----
    bracket_pat = re.compile(r"\[[^\]]+\]")
    two_letter_atoms = Counter()
    bracket_tokens = Counter()
    for s in train[: min(200_000, len(train))]:
        for tok in re.findall(r"Cl|Br|Si|Se", s):
            two_letter_atoms[tok] += 1
        for tok in bracket_pat.findall(s):
            bracket_tokens[tok] += 1
    print(f"\n2-letter atom tokens (first 200k): {two_letter_atoms}")
    print(f"Distinct bracket tokens (first 200k): {len(bracket_tokens)}")
    print("Top 20 bracket tokens:", bracket_tokens.most_common(20))

    # ---- RDKit-canonical subsample (heavy) ----
    SAMPLE_N = 20_000
    sample_idx = random.sample(range(len(train)), SAMPLE_N)
    sample_smi = [train[i] for i in sample_idx]

    print(f"\nProcessing random subsample of {SAMPLE_N:,} for RDKit-based stats...")

    valid = 0
    canon_equal_input = 0
    heavy_atoms = []
    rings = []
    aromatic_rings = []
    mw = []
    logp = []
    tpsa = []
    qed = []
    rotb = []
    fsp3 = []
    atom_counter: Counter = Counter()
    has_charged = 0
    has_stereo = 0
    scaffolds = Counter()

    for s in sample_smi:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        valid += 1
        can = Chem.MolToSmiles(mol)
        if can == s:
            canon_equal_input += 1
        heavy_atoms.append(mol.GetNumHeavyAtoms())
        rings.append(mol.GetRingInfo().NumRings())
        aromatic_rings.append(
            sum(1 for r in mol.GetRingInfo().AtomRings()
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))
        )
        mw.append(Descriptors.MolWt(mol))
        logp.append(Descriptors.MolLogP(mol))
        tpsa.append(Descriptors.TPSA(mol))
        try:
            qed.append(QED.qed(mol))
        except Exception:
            pass
        rotb.append(Descriptors.NumRotatableBonds(mol))
        fsp3.append(Descriptors.FractionCSP3(mol))
        for atom in mol.GetAtoms():
            atom_counter[atom.GetSymbol()] += 1
        if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
            has_charged += 1
        if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()):
            has_stereo += 1
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if scaf:
                scaffolds[scaf] += 1
        except Exception:
            pass

    print(
        f"\nValidity in subsample: {valid}/{SAMPLE_N} = {valid / SAMPLE_N:.4f}\n"
        f"Already canonical (string==MolToSmiles(MolFromSmiles)): "
        f"{canon_equal_input}/{valid} = {canon_equal_input / valid:.4f}"
    )

    def describe(name: str, arr: list) -> dict:
        a = np.asarray(arr, dtype=float)
        return {
            "name": name,
            "n": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "p05": float(np.percentile(a, 5)),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
        }

    prop_stats = {
        "heavy_atoms": describe("heavy_atoms", heavy_atoms),
        "n_rings": describe("n_rings", rings),
        "aromatic_rings": describe("aromatic_rings", aromatic_rings),
        "MolWt": describe("MolWt", mw),
        "MolLogP": describe("MolLogP", logp),
        "TPSA": describe("TPSA", tpsa),
        "QED": describe("QED", qed),
        "NumRotatableBonds": describe("NumRotatableBonds", rotb),
        "FractionCSP3": describe("FractionCSP3", fsp3),
    }
    for v in prop_stats.values():
        print(
            f"  {v['name']:>18}: mean={v['mean']:.3f} "
            f"median={v['median']:.3f} p95={v['p95']:.3f} max={v['max']:.3f}"
        )

    print(f"\nAtom symbol distribution (subsample, top 20):")
    for sym, cnt in atom_counter.most_common(20):
        print(f"  {sym:>4}: {cnt:,}")

    print(
        f"\nMolecules with formal charge: {has_charged}/{valid} = {has_charged / valid:.4f}"
    )
    print(
        f"Molecules with explicit stereo: {has_stereo}/{valid} = {has_stereo / valid:.4f}"
    )

    n_scaf = sum(scaffolds.values())
    print(
        f"\nMurcko scaffolds (subsample): unique={len(scaffolds):,} / "
        f"{n_scaf:,} mols = {len(scaffolds) / n_scaf:.4f} unique-ratio"
    )
    print("Top 10 scaffolds:")
    for s, c in scaffolds.most_common(10):
        print(f"  {c:>5}  {s}")

    # ---- Length histogram on full corpus ----
    plt.figure(figsize=(6, 3))
    plt.hist(lengths, bins=80)
    plt.xlabel("SMILES length")
    plt.ylabel("count")
    plt.title(f"SMILES length, n={len(train):,}")
    plt.tight_layout()
    plt.savefig(OUT / "length_hist.png", dpi=130)
    plt.close()

    # ---- Property histograms ----
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    panels = [
        ("MolWt", mw, np.arange(0, 800, 10)),
        ("MolLogP", logp, np.arange(-4, 10, 0.2)),
        ("TPSA", tpsa, np.arange(0, 250, 5)),
        ("QED", qed, np.arange(0, 1.01, 0.02)),
        ("NumHeavyAtoms", heavy_atoms, np.arange(0, 80, 1)),
        ("FractionCSP3", fsp3, np.arange(0, 1.01, 0.02)),
    ]
    for ax, (name, vals, bins) in zip(axes.flat, panels):
        ax.hist(vals, bins=bins)
        ax.set_title(name)
    fig.suptitle(f"Training-set physchem properties (n={valid:,} subsample)")
    fig.tight_layout()
    fig.savefig(OUT / "physchem_hist.png", dpi=130)
    plt.close(fig)

    # ---- Save JSON report ----
    report = {
        "n_train": len(train),
        "n_eval_train": len(eval_train),
        "user_train_eq_eval_train_set": same_set,
        "length": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": int(np.median(lengths)),
            "p95": int(np.percentile(lengths, 95)),
            "p99": int(np.percentile(lengths, 99)),
        },
        "subsample_n": SAMPLE_N,
        "validity_subsample": valid / SAMPLE_N,
        "already_canonical_fraction": canon_equal_input / valid,
        "fraction_charged": has_charged / valid,
        "fraction_stereo": has_stereo / valid,
        "scaffold_unique_ratio_subsample": len(scaffolds) / n_scaf,
        "top_atom_symbols": atom_counter.most_common(20),
        "top_two_letter_atom_tokens": two_letter_atoms.most_common(),
        "n_bracket_tokens_first200k": len(bracket_tokens),
        "top_bracket_tokens": bracket_tokens.most_common(20),
        "char_vocab_size": len(char_counter),
        "top_chars": char_counter.most_common(30),
        "properties": prop_stats,
    }
    with open(OUT / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport written to {OUT/'report.json'}")
    print(f"Plots written to {OUT}")


if __name__ == "__main__":
    main()

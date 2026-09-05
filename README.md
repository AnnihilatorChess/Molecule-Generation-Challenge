# Molecule Generation Challenge

JKU Linz "AI for Life Sciences" course challenge. Placed **2nd out of 143** participants.

## The task

Train a model on about 1.27 million real drug-like molecules (written as SMILES strings) and
generate 10,000 new ones that look like they could belong to the same set.

Quality was measured with **FCD** (Frechet ChemNet Distance): lower means the generated
molecules match the statistics of real molecules more closely. Three extra checks had to pass
first: at least 90% of generated molecules had to be valid, unique, and not copied from training
data.

## My approach

**1. Language model over SMILES.** I treated a molecule's SMILES string as a sequence of tokens
and trained a model to predict the next token, then generated new molecules one token at a time.

- Started with an LSTM, then switched to a Transformer, then scaled the Transformer up
  (more layers, more parameters). Each step lowered the FCD score. The final model was an
  8-layer Transformer with 25 million parameters.
- Bigger models kept helping up to a point, then flattened out. Once that happened, retraining
  further was not worth it and I moved to a different lever.

**2. Picking the best subset instead of retraining.** The score is computed on the 10,000
molecules you submit, not on everything the model can produce. So instead of generating exactly
10,000, I generated a much larger pool (up to 200,000 molecules) and then searched for the
10,000-molecule subset whose statistics best matched the target.

This search works by repeatedly swapping one molecule in the subset for one outside it, keeping
the swap only if it improves the score. Running this for thousands of swaps on a large pool gave
a much bigger improvement than any further model training would have.

This is a legitimate use of information the challenge organizers provided (the target statistics
were shipped with the evaluation code), but it is worth being honest that it is a search over
which molecules to submit, not a better generative model. The "how good is my model, really"
number is the score before this subset search, about 0.26 FCD. Full reasoning is in
[VERSION_HISTORY.md](VERSION_HISTORY.md).

## Result

| Stage | Official FCD |
|---|---|
| First submission (small Transformer) | 0.359 |
| Scaled-up Transformer | 0.258 |
| + subset search on a 50k pool | 0.157 |
| + subset search on a 200k pool | **0.083** (best) |

All submissions kept validity, uniqueness, and novelty at or near 1.0.

## Files

- `code/` - all model versions, from the first LSTM (`v1`) to the final pool-search pipeline (`v6`)
- `VERSION_HISTORY.md` - detailed log of every version, its results, and what I learned from it
- `CHALLENGE_PLAN.md` - the plan and data analysis written before starting
- `CHALLENGE_DESCRIPTION.txt` - the original task description
- `report.pdf` - final write-up submitted for grading
- `presentation.tex` - slides for the challenge presentation
- `predictions/best/` - the best submission files
- `notebooks/` - exploratory data analysis
- `metric_fcd.py` - the official evaluation script

## What is not included

The training data (1.27 million SMILES, provided by the course) and most intermediate prediction
files are left out to keep the repo small. Only the best submissions per stage are kept.

## Dependencies

PyTorch, RDKit, the `fcd` package (Frechet ChemNet Distance), numpy, pandas.

# Plan

Notes I wrote before starting, to think through the task and decide on an approach.

## Starting point

The course gave us a small example: a tiny RNN trained on a small, simple molecule dataset
(QM9), which reads and writes SMILES strings one character at a time. It got okay but not
great scores.

Our real dataset is much bigger and harder: 1.27 million molecules instead of 130,000, longer
and more complex SMILES strings, and a wider range of atoms and ring types. The idea from the
example still applies, but the model needs to be bigger and use a smarter way of splitting
SMILES into tokens (by chemical group, not by single character).

## What `test_stats.p` is

This file, given with the challenge, holds the exact statistics (mean and spread) that the
grading server computes from the hidden test set. It does not contain the actual test
molecules, just these summary numbers. This means the score can be understood and estimated
without ever seeing the real test set, since the same kind of statistics can be computed on a
held-out slice of the training data as a local stand-in.

## What I found looking at the data

- All 1.27 million training molecules are valid and already in a clean, standard format.
- Splitting SMILES by single character wastes information, since some atoms are written as two
  characters (like Cl or Br). A tokenizer that understands this gives shorter, cleaner
  sequences and faster learning.
- Typical molecule length is under 50 characters, max 100.
- The molecules are drug-like: many rings, moderate size, and quite structurally diverse (most
  molecules do not share a common core structure with others).

## Plan

1. Build a way to score my own models locally, using a held-out slice of training data, so I
   do not waste real submissions on ideas that will not work.
2. Start with a simple LSTM model, then move to a Transformer, and scale it up if it keeps
   helping.
3. Once training stops improving things, try picking a better subset of generated molecules to
   submit, instead of just training more.
4. Save a few submissions for late improvements once the main approach is working.

## What I decided to skip

- Reward-based fine-tuning towards a target property: not useful here since the goal is just
  to match the training data, not optimize a property.
- VAE or GAN-based generators: usually weaker than a good language model for this kind of task.
- Graph-based generators and diffusion models: more complex to build than a well-tuned language
  model, without a clear benefit at this scale.

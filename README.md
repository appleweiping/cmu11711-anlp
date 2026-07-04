# minBERT — CMU 11-711 Advanced NLP (Assignment 1)

> A from-skeleton implementation of **BERT** — multi-head self-attention, the
> Transformer encoder layer, embeddings, and a decoupled-weight-decay AdamW —
> plus sentence-classification fine-tuning and a multitask extension
> (sentiment / paraphrase / semantic similarity). Independent educational build
> of **CMU 11-711 Advanced NLP** (Graham Neubig), part of a
> [csdiy.wiki](https://csdiy.wiki/) full-catalog run.

![status](https://img.shields.io/badge/status-complete-brightgreen)
![language](https://img.shields.io/badge/python-informational)
![license](https://img.shields.io/badge/license-MIT-blue)

## Overview

CMU 11-711 Assignment 1 (**minBERT**) asks you to implement the core of BERT
from a skeleton — no `transformers` library allowed — then load the public
`bert-base-uncased` weights into your own modules and fine-tune for sentence
classification. This repo implements every `#todo` in the official
[`neubig/minbert-assignment`](https://github.com/neubig/minbert-assignment)
skeleton and verifies it against the course's own reference checks, then adds a
multitask fine-tuning module built on the *same* verified encoder.

What I implemented from scratch:

- **Scaled dot-product multi-head self-attention** (`BertSelfAttention.attention`)
- **The Transformer/BERT encoder layer** — residual *add-norm* around the
  attention and feed-forward sub-layers (`BertLayer.add_norm`, `BertLayer.forward`)
- **BERT embeddings** — word + positional + token-type (`BertModel.embed`)
- **AdamW** with decoupled weight decay and the "efficient" bias correction
  (`optimizer.py`)
- **The classification head** and fine-tune/pretrain pipeline (`classifier.py`)
- **A multitask head set** — sentiment (5-way), paraphrase (binary), semantic
  similarity (regression) — on the shared encoder (`multitask_classifier.py`)

## Results (measured on this machine: Windows, CPU-only, 3 threads)

Verification checks (the course's own):

| Check | Command | Result |
|---|---|---|
| BERT forward matches reference | `python sanity_check.py` | **`Your BERT implementation is correct!`** |
| AdamW matches reference | `python optimizer_test.py` | **`Optimizer test passed!`** |

Single-task sentiment classification (`classifier.py` components), measured on
this machine — frozen `bert-base-uncased`, one epoch over the **full** SST train
set (8,544 sentences), evaluated on the **full** SST dev set (1,101 sentences):

| Task | Setup | Train / Dev | Dev accuracy | Dev macro-F1 |
|---|---|---|---|---|
| SST-5 sentiment | `pretrain` (frozen BERT + linear head), 1 epoch, lr 1e-3, bs 16 | 8,544 / 1,101 | **0.3869** | 0.2501 |

For reference, the official assignment README reports SST `pretrain` dev
accuracy ≈ **0.391** (mean over 10 seeds); the **0.3869** measured here lands
right on that reference, confirming the frozen-encoder + head pipeline works.
Full fine-tuning of all ~110M BERT parameters is *much* heavier on CPU, so the
multitask experiment below (which does update the encoder) is deliberately
capped in size to stay reproducible on this hardware.

Multitask fine-tuning (`multitask_classifier.py`) — the shared minBERT encoder
is fine-tuned jointly on all three tasks, one epoch, 1,200 train / 500 dev
examples per task (225 optimizer steps), lr 1e-5, bs 16:

| Task | Metric | Measured (dev) | Random baseline |
|---|---|---|---|
| SST sentiment | accuracy (5-way) | **0.4620** | ~0.20 |
| Quora paraphrase | accuracy (binary) | **0.6220** | ~0.50 |
| STS similarity | Pearson correlation | **0.1544** | ~0.00 |

All three heads learn signal well above chance from a single short epoch; the
absolute numbers are modest because the run is intentionally small (CPU-only,
under concurrent load — the multitask epoch took ~39 min of wall clock). Scaling
`--max-train-per-task 0` (full data) and adding epochs on a GPU recovers the
paper-level accuracies.

Raw logs and model outputs are under [`results/`](results/) and [`logs/`](logs/).

## Implemented assignments

- [x] **Assignment 1 — Build Your Own BERT (minBERT)**
  - [x] Multi-head self-attention
  - [x] Transformer encoder layer (add-norm blocks, feed-forward)
  - [x] BERT embeddings (word + position + token-type)
  - [x] AdamW optimizer (decoupled weight decay)
  - [x] Sentence-classification head + pretrain/finetune pipeline
  - [x] Passes `sanity_check.py` and `optimizer_test.py`
  - [x] SST single-task run with measured dev accuracy (**0.3869**, matching the
    ~0.391 reference)
- [x] **Multitask extension** — sentiment (SST) + paraphrase (Quora) +
  semantic similarity (STS) on the shared minBERT encoder, with measured
  metrics (SST acc **0.4620** / Quora acc **0.6220** / STS Pearson **0.1544**)

> Assignments 2–4 of CMU 11-711 are open-ended *group* projects (an end-to-end
> NLP system, a state-of-the-art reimplementation, and a novel final project)
> rather than autograded skeletons, so they are out of scope for a single-repo,
> reproducible CPU build. This repo delivers the signature coding assignment
> (minBERT) in full plus its multitask extension.

## Project structure

```
cmu11711-anlp/
├── bert.py                     # minBERT: attention, encoder layer, embeddings  (implemented)
├── base_bert.py                # weight loading / from_pretrained (skeleton, HF-derived)
├── classifier.py               # single-task pretrain/finetune pipeline + head (implemented)
├── multitask_classifier.py     # sentiment + paraphrase + similarity heads      (implemented)
├── optimizer.py                # AdamW with decoupled weight decay              (implemented)
├── config.py, tokenizer.py, utils.py   # skeleton support (HF-derived; fnmatch bug fixed)
├── sanity_check.py / .data     # official BERT reference check
├── optimizer_test.py / .npy    # official AdamW reference check
├── download_multitask_data.py  # fetch SST/Quora/STS splits at runtime
├── data/                       # SST + CFIMDB (bundled); multitask CSVs (downloaded)
├── results/                    # measured outputs
└── logs/                       # training logs
```

## How to run

```bash
# Python 3.11 (shared csdiy env): D:\Project\_csdiy\.venv-ml\Scripts\python.exe
pip install -r requirements.txt

# --- Verification (the course's own checks) ---
python sanity_check.py       # -> "Your BERT implementation is correct!"
python optimizer_test.py     # -> "Optimizer test passed!"

# --- Reproduce the measured results in results/metrics.json (one process) ---
python download_multitask_data.py     # fetch the SST-ids / Quora / STS CSVs
python run_results.py all              # SST single-task + multitask, CPU
# (or run a single experiment:  python run_results.py sst  /  ...multitask)

# --- Or drive the graded scripts directly ---
# SST single-task, frozen BERT (pretrain) — only the classifier head trains:
python classifier.py --option pretrain --epochs 1 --lr 1e-3 --batch_size 16 \
    --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt
# SST / CFIMDB, full fine-tuning (heavy on CPU — smaller is realistic):
python classifier.py --option finetune --epochs 2 --lr 1e-5 \
    --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt

# Multitask fine-tuning (sentiment / paraphrase / similarity):
python multitask_classifier.py --option finetune --epochs 1 --lr 1e-5 \
    --max-train-per-task 1200 --max-eval-per-task 500
```

Notes for CPU / restricted networks:
- `bert-base-uncased` weights and vocab are fetched from HuggingFace on first
  use and cached under `~/.cache/huggingface/transformers`. If `huggingface.co`
  is unreliable, set `HF_ENDPOINT=https://hf-mirror.com`, or pre-cache the
  files (the repo already fixes the skeleton's `fnmatch` cache bug so the
  offline fallback works).
- `CSDIY_NUM_THREADS` (default 3) caps CPU threads.

## Verification

- **BERT correctness:** `sanity_check.py` reloads reference embeddings computed
  by the assignment authors and asserts our `BertModel` output matches to
  `atol=1e-5`. It prints `Your BERT implementation is correct!`.
- **Optimizer correctness:** `optimizer_test.py` trains a tiny linear model for
  1000 steps and asserts our `AdamW` reproduces the reference weights exactly.
  It prints `Optimizer test passed!`.
- **Real training runs:** the accuracies in the results tables come from actually
  running `classifier.py` / `multitask_classifier.py` on this machine; the logs
  are saved under `logs/` and the per-example predictions under `results/`.

## Tech stack

Python 3.11, PyTorch (CPU), NumPy, scikit-learn, HuggingFace `tokenizers`
(WordPiece only — the model itself is implemented from scratch). No
`transformers` library is used for the model, per the assignment rules.

## Key ideas / what I learned

- Multi-head attention is just a batched scaled dot product with a per-head
  reshape; the padding mask is added *before* the softmax as a large negative
  bias so padded keys get ~zero weight.
- A Transformer block is two residual *add-norm* sub-layers; getting the residual
  connection (add the sub-layer *input*, not the projected output) right is what
  makes the pretrained weights load and reproduce the reference exactly.
- BERT's input embedding is the sum of word, absolute-position, and token-type
  embeddings, then LayerNorm + dropout.
- AdamW decouples weight decay from the gradient step; the "efficient" bias
  correction folds `sqrt(1-β₂ᵗ)/(1-β₁ᵗ)` into the step size instead of forming
  `m̂`, `v̂` explicitly.
- For sentence-pair tasks, `[u; v; |u−v|; u·v]` over the two `[CLS]` vectors is a
  strong, cheap interaction feature for both classification and regression heads.

## Credits & license

Based on **Assignment 1 (minBERT)** of **CMU CS 11-711 Advanced NLP** by
Graham Neubig and TAs (Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt,
Aditya Veerubhotla). The skeleton support files (`base_bert.py`, `tokenizer.py`,
`utils.py`, `config.py`) derive from the HuggingFace
[`transformers`](https://github.com/huggingface/transformers) library
(Apache-2.0). The multitask data splits (SST / Quora / STS) belong to their
original authors and are downloaded at runtime, not redistributed here.

This repository is an independent educational reimplementation. Original code I
wrote (the minBERT model internals, optimizer, classifier heads, and multitask
pipeline) is released under the [MIT License](LICENSE). Course materials,
datasets, and pretrained weights remain the property of their respective owners.

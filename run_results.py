"""Produce the measured results for the README, in ONE CPU process.

Two experiments, both reusing the *graded* components verified by
``sanity_check.py`` (BERT) and ``optimizer_test.py`` (AdamW):

  1. Single-task SST sentiment classifier (Assignment 1 style).
     Frozen BERT ("pretrain" option) + a trainable linear head over the
     [CLS] pooled output. Reports dev accuracy on the *full* SST dev set.

  2. Multitask setup (Assignment 2 style): SST sentiment + Quora paraphrase
     + STS similarity, fine-tuning the shared BERT encoder at a modest-but-
     real scale (capped #examples/task so it finishes on CPU). Reports dev
     accuracy (SST, Quora) and Pearson correlation (STS).

Results are written incrementally to results/metrics.json so partial progress
always survives. Everything runs on CPU with OMP_NUM_THREADS / CSDIY_NUM_THREADS
kept modest.
"""
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.set_num_threads(int(os.environ.get("CSDIY_NUM_THREADS", "3")))

import classifier as C
import multitask_classifier as M

RESULTS = os.path.join("results", "metrics.json")
os.makedirs("results", exist_ok=True)


def save(metrics):
    with open(RESULTS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {RESULTS}", flush=True)


def load():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {}


def eval_head(dataloader, model, device):
    """Efficient dev evaluation: accuracy + macro-F1 (no train-set pass)."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for b in dataloader:
            logits = model(b["token_ids"].to(device), b["attention_mask"].to(device))
            y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
            y_true.extend(b["labels"].view(-1).tolist())
    from sklearn.metrics import accuracy_score, f1_score
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def single_task_sst(metrics, epochs, lr, batch_size):
    device = torch.device("cpu")
    train_data, num_labels = C.create_data("data/sst-train.txt", "train")
    dev_data = C.create_data("data/sst-dev.txt", "valid")

    ds_args = SimpleNamespace(hidden_dropout_prob=0.3)
    train_ds = C.BertDataset(train_data, ds_args)
    dev_ds = C.BertDataset(dev_data, ds_args)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=train_ds.collate_fn)
    dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=batch_size, collate_fn=dev_ds.collate_fn)

    config = SimpleNamespace(hidden_dropout_prob=0.3, num_labels=num_labels,
                             hidden_size=768, data_dir=".", option="pretrain")
    model = C.BertSentClassifier(config).to(device)
    opt = C.AdamW(model.parameters(), lr=lr)

    best = 0.0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for b in train_dl:
            opt.zero_grad()
            logits = model(b["token_ids"].to(device), b["attention_mask"].to(device))
            loss = F.nll_loss(logits, b["labels"].view(-1).to(device), reduction="sum") / batch_size
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        dev_acc, dev_f1 = eval_head(dev_dl, model, device)
        best = max(best, dev_acc)
        print(f"[sst_pretrain] epoch {ep}: loss={tot/nb:.3f} dev_acc={dev_acc:.4f} "
              f"dev_f1={dev_f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        metrics["sst_single_task_pretrain"] = {
            "task": "SST-5 sentiment (single task)",
            "option": "pretrain (frozen BERT + linear head)",
            "epochs_done": ep + 1, "epochs_total": epochs,
            "lr": lr, "batch_size": batch_size,
            "train_size": len(train_data), "dev_size": len(dev_data),
            "dev_acc": round(dev_acc, 4), "best_dev_acc": round(best, 4),
            "dev_macro_f1": round(dev_f1, 4),
        }
        save(metrics)
    print(f"[sst_pretrain] DONE best_dev_acc={best:.4f}", flush=True)


def multitask(metrics, epochs, lr, batch_size, max_train, max_eval):
    M.seed_everything(11711)
    args = SimpleNamespace(
        option="finetune", epochs=epochs, lr=lr, batch_size=batch_size,
        hidden_dropout_prob=0.3, seed=11711, use_gpu=False,
        max_train_per_task=max_train, max_eval_per_task=max_eval, max_steps=0,
        sst_train="data/ids-sst-train.csv", sst_dev="data/ids-sst-dev.csv",
        para_train="data/quora-train.csv", para_dev="data/quora-dev.csv",
        sts_train="data/sts-train.csv", sts_dev="data/sts-dev.csv",
    )
    t0 = time.time()
    sst_acc, para_acc, sts_corr = M.train(args)
    metrics["multitask_finetune"] = {
        "tasks": "SST sentiment + Quora paraphrase + STS similarity",
        "option": "finetune (shared BERT encoder updated)",
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "max_train_per_task": max_train, "max_eval_per_task": max_eval,
        "sst_dev_acc": round(sst_acc, 4),
        "quora_dev_acc": round(para_acc, 4),
        "sts_dev_pearson": round(sts_corr, 4),
        "wall_clock_sec": round(time.time() - t0, 0),
    }
    save(metrics)
    print(f"[multitask] DONE sst={sst_acc:.4f} quora={para_acc:.4f} sts_pearson={sts_corr:.4f}", flush=True)


def main():
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    metrics = load()
    C.seed_everything(11711)
    if which in ("all", "sst"):
        # 1 epoch of frozen-BERT head training over the full SST train set is a
        # real, reference-comparable result on CPU (see results/metrics.json).
        single_task_sst(metrics, epochs=1, lr=1e-3, batch_size=16)
    if which in ("all", "multitask"):
        multitask(metrics, epochs=1, lr=1e-5, batch_size=16, max_train=1200, max_eval=500)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

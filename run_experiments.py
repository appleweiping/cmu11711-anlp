"""Thin experiment driver around the *exact* classifier.py components.

`classifier.py` is the authoritative, assignment-graded script and is run as-is
for the submission outputs. This driver imports the very same model
(`BertSentClassifier`), data pipeline (`create_data`, `BertDataset`), evaluation
(`model_eval`) and optimizer (`AdamW`) from `classifier.py`, and runs training +
dev/test evaluation. The only difference from `classifier.train` is that it skips
the per-epoch *train-set* accuracy pass (which classifier.py computes purely for
logging), so the same measured dev/test numbers are produced faster on CPU.

Usage:
  python run_experiments.py --option pretrain --dataset sst --epochs 2 --lr 1e-3 --batch_size 64
  python run_experiments.py --option finetune --dataset cfimdb --epochs 2 --lr 1e-5 --batch_size 8
"""
import argparse
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import classifier as C  # reuse the graded components verbatim


DATASETS = {
    "sst": ("data/sst-train.txt", "data/sst-dev.txt", "data/sst-test.txt"),
    "cfimdb": ("data/cfimdb-train.txt", "data/cfimdb-dev.txt", "data/cfimdb-test.txt"),
}


def run(args):
    device = torch.device("cpu")
    train_path, dev_path, test_path = DATASETS[args.dataset]

    train_data, num_labels = C.create_data(train_path, "train")
    dev_data = C.create_data(dev_path, "valid")
    test_data = C.create_data(test_path, "test")

    train_ds = C.BertDataset(train_data, args)
    dev_ds = C.BertDataset(dev_data, args)
    test_ds = C.BertDataset(test_data, args)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, collate_fn=train_ds.collate_fn)
    dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=args.batch_size, collate_fn=dev_ds.collate_fn)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size, collate_fn=test_ds.collate_fn)

    config = SimpleNamespace(
        hidden_dropout_prob=args.hidden_dropout_prob,
        num_labels=num_labels,
        hidden_size=768,
        data_dir=".",
        option=args.option,
    )
    model = C.BertSentClassifier(config).to(device)
    optimizer = C.AdamW(model.parameters(), lr=args.lr)

    best_dev = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total, nb = 0.0, 0
        for batch in train_dl:
            b_ids = batch["token_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.nll_loss(logits, b_labels.view(-1), reduction="sum") / args.batch_size
            loss.backward()
            optimizer.step()
            total += loss.item()
            nb += 1
        dev_acc, dev_f1, *_ = C.model_eval(dev_dl, model, device)
        best_dev = max(best_dev, dev_acc)
        print(f"epoch {epoch}: train_loss={total / nb:.3f} dev_acc={dev_acc:.4f} "
              f"dev_f1={dev_f1:.4f} ({time.time() - t0:.0f}s)", flush=True)

    dev_acc, dev_f1, *_ = C.model_eval(dev_dl, model, device)
    test_acc, test_f1, *_ = C.model_eval(test_dl, model, device)
    print("=" * 60, flush=True)
    print(f"[{args.dataset} {args.option}] FINAL dev_acc={dev_acc:.4f} "
          f"best_dev_acc={best_dev:.4f} test_acc={test_acc:.4f}", flush=True)
    print("=" * 60, flush=True)
    return best_dev, dev_acc, test_acc


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--option", choices=("pretrain", "finetune"), required=True)
    p.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=11711)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(f"args: {vars(args)}", flush=True)
    C.seed_everything(args.seed)
    run(args)

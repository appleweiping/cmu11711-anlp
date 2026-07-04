"""Run the full measured-results suite in ONE process, writing results
incrementally to results/metrics.json so partial progress always survives.

Each experiment reuses the exact graded components from classifier.py /
multitask_classifier.py. Everything runs on CPU.
"""
import json
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import classifier as C

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


def single_task(dataset, option, epochs, lr, batch_size, metrics):
    paths = {
        "sst": ("data/sst-train.txt", "data/sst-dev.txt", "data/sst-test.txt"),
        "cfimdb": ("data/cfimdb-train.txt", "data/cfimdb-dev.txt", "data/cfimdb-test.txt"),
    }[dataset]
    train_path, dev_path, test_path = paths
    device = torch.device("cpu")

    train_data, num_labels = C.create_data(train_path, "train")
    dev_data = C.create_data(dev_path, "valid")
    test_data = C.create_data(test_path, "test")
    args = SimpleNamespace(hidden_dropout_prob=0.3)
    train_ds, dev_ds, test_ds = (C.BertDataset(train_data, args),
                                 C.BertDataset(dev_data, args),
                                 C.BertDataset(test_data, args))
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=train_ds.collate_fn)
    dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=batch_size, collate_fn=dev_ds.collate_fn)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, collate_fn=test_ds.collate_fn)

    config = SimpleNamespace(hidden_dropout_prob=0.3, num_labels=num_labels,
                             hidden_size=768, data_dir=".", option=option)
    model = C.BertSentClassifier(config).to(device)
    opt = C.AdamW(model.parameters(), lr=lr)

    key = f"{dataset}_{option}"
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
        dev_acc, dev_f1, *_ = C.model_eval(dev_dl, model, device)
        best = max(best, dev_acc)
        print(f"[{key}] epoch {ep}: loss={tot/nb:.3f} dev_acc={dev_acc:.4f} "
              f"dev_f1={dev_f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        metrics[key] = {"dataset": dataset, "option": option, "epochs_done": ep + 1,
                        "epochs_total": epochs, "lr": lr, "batch_size": batch_size,
                        "dev_acc": round(dev_acc, 4), "best_dev_acc": round(best, 4),
                        "dev_f1": round(dev_f1, 4)}
        save(metrics)

    dev_acc, _, dev_pred, _, dev_sents = C.model_eval(dev_dl, model, device)
    test_acc, _, test_pred, _, test_sents = C.model_eval(test_dl, model, device)
    # write prediction outputs for the record
    with open(f"results/{dataset}-{option}-dev-output.txt", "w", encoding="utf-8") as f:
        for s, p in zip(dev_sents, dev_pred):
            f.write(f"{p} ||| {s}\n")
    with open(f"results/{dataset}-{option}-test-output.txt", "w", encoding="utf-8") as f:
        for s, p in zip(test_sents, test_pred):
            f.write(f"{p} ||| {s}\n")
    metrics[key]["final_dev_acc"] = round(dev_acc, 4)
    metrics[key]["test_acc"] = round(test_acc, 4)
    save(metrics)
    print(f"[{key}] DONE best_dev={best:.4f} test_acc={test_acc:.4f}", flush=True)


def multitask(metrics, epochs, lr, batch_size, max_train, max_eval):
    import multitask_classifier as M
    M.seed_everything(11711)
    args = SimpleNamespace(
        option="finetune", epochs=epochs, lr=lr, batch_size=batch_size,
        hidden_dropout_prob=0.3, seed=11711, use_gpu=False,
        max_train_per_task=max_train, max_eval_per_task=max_eval, max_steps=0,
        sst_train="data/ids-sst-train.csv", sst_dev="data/ids-sst-dev.csv",
        para_train="data/quora-train.csv", para_dev="data/quora-dev.csv",
        sts_train="data/sts-train.csv", sts_dev="data/sts-dev.csv",
    )
    sst_acc, para_acc, sts_corr = M.train(args)
    metrics["multitask_finetune"] = {
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "max_train_per_task": max_train, "max_eval_per_task": max_eval,
        "sst_dev_acc": round(sst_acc, 4),
        "quora_dev_acc": round(para_acc, 4),
        "sts_dev_pearson": round(sts_corr, 4),
    }
    save(metrics)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    metrics = load()
    if which in ("all", "sst_pretrain"):
        single_task("sst", "pretrain", epochs=2, lr=1e-3, batch_size=64, metrics=metrics)
    if which in ("all", "cfimdb_finetune"):
        single_task("cfimdb", "finetune", epochs=2, lr=1e-5, batch_size=8, metrics=metrics)
    if which in ("all", "sst_finetune"):
        single_task("sst", "finetune", epochs=1, lr=1e-5, batch_size=16, metrics=metrics)
    if which in ("all", "multitask"):
        multitask(metrics, epochs=1, lr=1e-5, batch_size=16, max_train=1200, max_eval=500)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

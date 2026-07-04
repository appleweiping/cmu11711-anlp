"""Multitask fine-tuning of minBERT (CMU 11-711 Advanced NLP).

This extends the single-task minBERT (Assignment 1) to the three sentence-level
tasks used in the CS11-711 / CS224N multitask minBERT project:

  * Sentiment analysis    (SST, 5 classes)     -> accuracy
  * Paraphrase detection  (Quora, binary)       -> accuracy
  * Semantic similarity   (STS, regression 0-5) -> Pearson correlation

It reuses the *same* BERT implementation from ``bert.py`` (verified against the
official ``sanity_check.py``) and the *same* ``AdamW`` from ``optimizer.py``
(verified against ``optimizer_test.py``). Only the task heads and the training
loop are new here.

Design for a CPU-only machine: fine-tuning all 110M BERT parameters is expensive,
so this script exposes ``--max-train-per-task`` / ``--max-eval-per-task`` /
``--max-steps`` to run a *modest but real* fine-tune and report genuine measured
metrics. Set them to 0 for the full datasets on a bigger machine.

Heads (all operate on the [CLS] pooled output of the shared encoder):
  * predict_sentiment(ids, mask)                      -> [bs, 5] logits
  * predict_paraphrase(ids1, mask1, ids2, mask2)      -> [bs] logit
  * predict_similarity(ids1, mask1, ids2, mask2)      -> [bs] similarity in [0, 5]

Run:
  python download_multitask_data.py
  python multitask_classifier.py --option finetune --epochs 1 \
      --max-train-per-task 1500 --max-eval-per-task 600 --lr 1e-5
"""
import argparse
import csv
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW

torch.set_num_threads(int(os.environ.get("CSDIY_NUM_THREADS", "3")))

BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


def seed_everything(seed: int = 11711) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def _read_csv(path):
    with open(path, "r", encoding="utf-8") as fp:
        return list(csv.DictReader(fp, delimiter="\t"))


def load_sentiment(path, limit=0):
    rows = _read_csv(path)
    data = [(r["sentence"].strip(), int(r["sentiment"].strip())) for r in rows]
    if limit:
        data = data[:limit]
    return data


def load_pairs(path, label_key, cast, limit=0):
    rows = _read_csv(path)
    data = []
    for r in rows:
        s1 = (r.get("sentence1") or "").strip()
        s2 = (r.get("sentence2") or "").strip()
        raw = (r.get(label_key) or "").strip()
        if raw == "":
            continue
        data.append((s1, s2, cast(raw)))
    if limit:
        data = data[:limit]
    return data


class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, batch):
        sents = [x[0] for x in batch]
        labels = torch.LongTensor([x[1] for x in batch])
        enc = self.tokenizer(sents, return_tensors="pt", padding=True, truncation=True)
        return {
            "token_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


class PairDataset(Dataset):
    def __init__(self, data, tokenizer, label_dtype):
        self.data = data
        self.tokenizer = tokenizer
        self.label_dtype = label_dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, batch):
        s1 = [x[0] for x in batch]
        s2 = [x[1] for x in batch]
        if self.label_dtype == "float":
            labels = torch.FloatTensor([x[2] for x in batch])
        else:
            labels = torch.LongTensor([x[2] for x in batch])
        e1 = self.tokenizer(s1, return_tensors="pt", padding=True, truncation=True)
        e2 = self.tokenizer(s2, return_tensors="pt", padding=True, truncation=True)
        return {
            "token_ids_1": e1["input_ids"],
            "attention_mask_1": e1["attention_mask"],
            "token_ids_2": e2["input_ids"],
            "attention_mask_2": e2["attention_mask"],
            "labels": labels,
        }


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
class MultitaskBERT(torch.nn.Module):
    """Shared minBERT encoder with three task-specific heads."""

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = config.option == "finetune"

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # sentiment: 5-way classification over a single sentence's [CLS]
        self.sentiment_head = torch.nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        # paraphrase: binary logit over [u; v; |u-v|; u*v] of the two sentences
        self.paraphrase_head = torch.nn.Linear(BERT_HIDDEN_SIZE * 4, 1)
        # similarity: scalar over the same pair features, squashed to [0, 5]
        self.similarity_head = torch.nn.Linear(BERT_HIDDEN_SIZE * 4, 1)

    def encode(self, input_ids, attention_mask):
        """Return the pooled [CLS] representation of a batch of sentences."""
        out = self.bert(input_ids, attention_mask)
        return out["pooler_output"]

    def _pair_features(self, ids1, mask1, ids2, mask2):
        u = self.dropout(self.encode(ids1, mask1))
        v = self.dropout(self.encode(ids2, mask2))
        return torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)

    def predict_sentiment(self, input_ids, attention_mask):
        pooled = self.dropout(self.encode(input_ids, attention_mask))
        return self.sentiment_head(pooled)

    def predict_paraphrase(self, ids1, mask1, ids2, mask2):
        feats = self._pair_features(ids1, mask1, ids2, mask2)
        return self.paraphrase_head(feats).squeeze(-1)

    def predict_similarity(self, ids1, mask1, ids2, mask2):
        feats = self._pair_features(ids1, mask1, ids2, mask2)
        # squash to the STS label range [0, 5]
        return torch.sigmoid(self.similarity_head(feats).squeeze(-1)) * 5.0


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_sentiment(loader, model, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        logits = model.predict_sentiment(
            batch["token_ids"].to(device), batch["attention_mask"].to(device)
        )
        y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
        y_true.extend(batch["labels"].tolist())
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return acc


@torch.no_grad()
def eval_paraphrase(loader, model, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        logit = model.predict_paraphrase(
            batch["token_ids_1"].to(device), batch["attention_mask_1"].to(device),
            batch["token_ids_2"].to(device), batch["attention_mask_2"].to(device),
        )
        y_pred.extend((torch.sigmoid(logit) > 0.5).long().cpu().tolist())
        y_true.extend([int(x) for x in batch["labels"].tolist()])
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return acc


@torch.no_grad()
def eval_similarity(loader, model, device):
    model.eval()
    preds, golds = [], []
    for batch in loader:
        pred = model.predict_similarity(
            batch["token_ids_1"].to(device), batch["attention_mask_1"].to(device),
            batch["token_ids_2"].to(device), batch["attention_mask_2"].to(device),
        )
        preds.extend(pred.cpu().tolist())
        golds.extend(batch["labels"].tolist())
    preds, golds = np.array(preds), np.array(golds)
    if preds.std() < 1e-8 or golds.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(preds, golds)[0, 1])


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate
    )


def train(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    ltr = args.max_train_per_task
    lev = args.max_eval_per_task

    sst_train = SentimentDataset(load_sentiment(args.sst_train, ltr), tokenizer)
    sst_dev = SentimentDataset(load_sentiment(args.sst_dev, lev), tokenizer)
    para_train = PairDataset(load_pairs(args.para_train, "is_duplicate", lambda x: int(float(x)), ltr), tokenizer, "long")
    para_dev = PairDataset(load_pairs(args.para_dev, "is_duplicate", lambda x: int(float(x)), lev), tokenizer, "long")
    sts_train = PairDataset(load_pairs(args.sts_train, "similarity", float, ltr), tokenizer, "float")
    sts_dev = PairDataset(load_pairs(args.sts_dev, "similarity", float, lev), tokenizer, "float")

    print(f"train sizes  -> sst {len(sst_train)}  para {len(para_train)}  sts {len(sts_train)}")
    print(f"dev   sizes  -> sst {len(sst_dev)}  para {len(para_dev)}  sts {len(sts_dev)}")

    sst_loader = make_loader(sst_train, args.batch_size, True)
    para_loader = make_loader(para_train, args.batch_size, True)
    sts_loader = make_loader(sts_train, args.batch_size, True)

    sst_dev_loader = make_loader(sst_dev, args.batch_size, False)
    para_dev_loader = make_loader(para_dev, args.batch_size, False)
    sts_dev_loader = make_loader(sts_dev, args.batch_size, False)

    config = SimpleNamespace(
        hidden_dropout_prob=args.hidden_dropout_prob,
        hidden_size=BERT_HIDDEN_SIZE,
        option=args.option,
    )
    model = MultitaskBERT(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        iters = [iter(sst_loader), iter(para_loader), iter(sts_loader)]
        task_names = ["sst", "para", "sts"]
        running = {t: 0.0 for t in task_names}
        counts = {t: 0 for t in task_names}
        step = 0
        active = [True, True, True]
        while any(active):
            for ti, name in enumerate(task_names):
                if not active[ti]:
                    continue
                try:
                    batch = next(iters[ti])
                except StopIteration:
                    active[ti] = False
                    continue
                optimizer.zero_grad()
                if name == "sst":
                    logits = model.predict_sentiment(
                        batch["token_ids"].to(device), batch["attention_mask"].to(device)
                    )
                    loss = F.cross_entropy(logits, batch["labels"].to(device))
                elif name == "para":
                    logit = model.predict_paraphrase(
                        batch["token_ids_1"].to(device), batch["attention_mask_1"].to(device),
                        batch["token_ids_2"].to(device), batch["attention_mask_2"].to(device),
                    )
                    loss = F.binary_cross_entropy_with_logits(
                        logit, batch["labels"].float().to(device)
                    )
                else:  # sts
                    pred = model.predict_similarity(
                        batch["token_ids_1"].to(device), batch["attention_mask_1"].to(device),
                        batch["token_ids_2"].to(device), batch["attention_mask_2"].to(device),
                    )
                    loss = F.mse_loss(pred, batch["labels"].to(device))
                loss.backward()
                optimizer.step()
                running[name] += loss.item()
                counts[name] += 1
                step += 1
                if args.max_steps and step >= args.max_steps:
                    active = [False, False, False]
                    break

        avg = {t: (running[t] / counts[t] if counts[t] else float("nan")) for t in task_names}
        print(f"epoch {epoch}: steps={step}  loss[sst]={avg['sst']:.3f} "
              f"loss[para]={avg['para']:.3f} loss[sts]={avg['sts']:.3f}")

    sst_acc = eval_sentiment(sst_dev_loader, model, device)
    para_acc = eval_paraphrase(para_dev_loader, model, device)
    sts_corr = eval_similarity(sts_dev_loader, model, device)

    print("=" * 60)
    print(f"DEV sentiment (SST) accuracy      : {sst_acc:.4f}")
    print(f"DEV paraphrase (Quora) accuracy   : {para_acc:.4f}")
    print(f"DEV similarity (STS) Pearson corr : {sts_corr:.4f}")
    print("=" * 60)
    return sst_acc, para_acc, sts_corr


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--option", choices=("pretrain", "finetune"), default="finetune")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=11711)
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--max-train-per-task", dest="max_train_per_task", type=int, default=0,
                   help="cap #train examples per task (0 = full dataset)")
    p.add_argument("--max-eval-per-task", dest="max_eval_per_task", type=int, default=0,
                   help="cap #dev examples per task (0 = full dev set)")
    p.add_argument("--max-steps", dest="max_steps", type=int, default=0,
                   help="global cap on optimizer steps per epoch (0 = no cap)")
    p.add_argument("--sst_train", default="data/ids-sst-train.csv")
    p.add_argument("--sst_dev", default="data/ids-sst-dev.csv")
    p.add_argument("--para_train", default="data/quora-train.csv")
    p.add_argument("--para_dev", default="data/quora-dev.csv")
    p.add_argument("--sts_train", default="data/sts-train.csv")
    p.add_argument("--sts_dev", default="data/sts-dev.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(f"args: {vars(args)}")
    seed_everything(args.seed)
    train(args)

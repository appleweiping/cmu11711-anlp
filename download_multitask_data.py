"""Download the multitask (SST / Quora / STS) datasets used by the extended
minBERT tasks (CMU 11-711 Advanced NLP, multitask variant of Assignment 1).

These CSVs are the standard splits distributed with the CS11-711 / CS224N minBERT
default project. They are NOT committed to this repository (they belong to their
original owners); this script fetches them at runtime into ``data/`` where
``multitask_classifier.py`` expects them.

Usage:
    python download_multitask_data.py
"""
import os
import sys
import requests

# Mirror-friendly raw GitHub source for the canonical minBERT multitask splits.
BASE = os.environ.get(
    "MULTITASK_DATA_URL",
    "https://raw.githubusercontent.com/gpoesia/minbert-default-final-project/main/data/",
)

FILES = [
    "ids-sst-train.csv",
    "ids-sst-dev.csv",
    "quora-train.csv",
    "quora-dev.csv",
    "sts-train.csv",
    "sts-dev.csv",
]


def download(fname: str, out_dir: str) -> None:
    dest = os.path.join(out_dir, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {fname} already present ({os.path.getsize(dest)} bytes)")
        return
    url = BASE + fname
    print(f"[get ] {url}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        tmp = dest + ".tmp"
        n = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1024 * 256):
                f.write(chunk)
                n += len(chunk)
        os.replace(tmp, dest)
    print(f"[ok  ] {fname} ({n} bytes)")


def main() -> int:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    for fname in FILES:
        download(fname, out_dir)
    print("All multitask datasets are in ./data")
    return 0


if __name__ == "__main__":
    sys.exit(main())

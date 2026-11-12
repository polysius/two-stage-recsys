"""Download the Amazon Reviews 2023 5-core ratings for one category and verify its checksum.

Usage: python scripts/download_data.py [--category Video_Games] [--data-dir data]
"""

import argparse
import hashlib
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "benchmark/5core/rating_only/{category}.csv"
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Video_Games")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    target = Path(args.data_dir) / "raw" / f"{args.category}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve(BASE_URL.format(category=args.category), target)

    checksum = sha256_of(target)
    (target.parent / f"{args.category}.sha256").write_text(f"{checksum}  {target.name}\n")
    print(f"{target} sha256={checksum}")


if __name__ == "__main__":
    main()

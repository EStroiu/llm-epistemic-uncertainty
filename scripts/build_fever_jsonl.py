import argparse
import json
import random
from typing import Dict, List

from datasets import load_dataset


LABEL_MAP: Dict[str, str] = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
}


def build_subset(output_path: str, max_examples: int, seed: int) -> int:
    ds = load_dataset(
        "parquet",
        data_files={"data": "hf://datasets/fever/fever@refs/convert/parquet/v1.0/labelled_dev/*.parquet"},
        split="data",
    )

    idxs = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(idxs)

    seen = set()
    rows: List[Dict[str, str]] = []
    for i in idxs:
        row = ds[i]
        claim_id = int(row["id"])
        if claim_id in seen:
            continue

        label = LABEL_MAP.get(str(row.get("label", "")).upper().strip())
        claim = str(row.get("claim", "")).strip()
        if not label or not claim:
            continue

        seen.add(claim_id)
        rows.append({"claim": claim, "label": label})
        if len(rows) >= max_examples:
            break

    with open(output_path, "w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FEVER subset JSONL for uncertainty benchmark.")
    parser.add_argument("--output-path", default="data/fever_labelled_dev_100.jsonl")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    count = build_subset(args.output_path, args.max_examples, args.seed)
    print(f"Wrote {count} rows to {args.output_path}")


if __name__ == "__main__":
    main()


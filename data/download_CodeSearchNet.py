from datasets import load_dataset
import json
from pathlib import Path

ds = load_dataset("claudios/code_search_net", "python")
out_dir = Path("data/raw/CodeSearchNet_python")
out_dir.mkdir(parents=True, exist_ok=True)

split_map = {
    "train": "train",
    "validation": "valid",
    "test": "test",
}

for hf_split, output_name in split_map.items():
    out_path = out_dir / f"{output_name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds[hf_split]:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

print(f"Saved to {out_dir}")

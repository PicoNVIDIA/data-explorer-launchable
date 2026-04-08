"""Generate file_structures.json cache for the DABStep data directory."""

import csv
import json
import os

DATA_DIR = "data/context"
OUTPUT = os.path.join(DATA_DIR, "file_structures.json")


def scan_csv(path: str) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    return {"file_type": "csv", "sample_row": row or {}}


def scan_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and data:
        sample = data[0]
        return {
            "file_type": "json",
            "structure_type": "array",
            "keys": list(sample.keys()) if isinstance(sample, dict) else [],
            "sample_record": sample if isinstance(sample, dict) else {},
        }
    elif isinstance(data, dict):
        return {
            "file_type": "json",
            "structure_type": "object",
            "keys": list(data.keys()),
            "sample_record": {},
        }
    return {"file_type": "json", "structure_type": "unknown"}


def main():
    structures = {}
    for fname in sorted(os.listdir(DATA_DIR)):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            if fname.endswith(".csv"):
                structures[fname] = scan_csv(fpath)
            elif fname.endswith(".json"):
                structures[fname] = scan_json(fpath)
        except Exception as e:
            structures[fname] = {"error": str(e)}

    with open(OUTPUT, "w") as f:
        json.dump(structures, f, indent=2, default=str)
    print(f"Generated {OUTPUT} with {len(structures)} entries")


if __name__ == "__main__":
    main()

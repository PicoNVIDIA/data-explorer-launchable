import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


DATA_DIR = "./data"
REPO_ID = "adyen/DABstep"
CONTEXT_PATTERN = "data/context/*"
TASKS_FILE = "tasks.json"


def download_dabstep_data(data_dir: str = DATA_DIR) -> None:
    """Download DABstep context data and tasks if they don't already exist."""
    data_path = Path(data_dir)

    # Check if context data already exists
    expected_files = [
        "context/payments.csv",
        "context/fees.json",
        "context/manual.md",
        "context/acquirer_countries.csv",
        "context/merchant_category_codes.csv",
        "context/merchant_data.json",
        "context/payments-readme.md",
    ]

    context_exists = all((data_path / f).exists() for f in expected_files)
    tasks_exist = (data_path / TASKS_FILE).exists()

    if context_exists and tasks_exist:
        print(f"DABstep data already exists in {data_dir}")
        return

    # Download context files if needed
    if not context_exists:
        print(f"Downloading DABstep context data to {data_dir}...")
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            allow_patterns=CONTEXT_PATTERN,
            local_dir=".",
        )
        print("Context download complete.")

    # Download and save tasks if needed
    if not tasks_exist:
        print(f"Downloading DABstep tasks to {data_dir}/{TASKS_FILE}...")
        data_path.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(REPO_ID, "tasks", split="default")
        tasks = [dict(row) for row in ds]
        with open(data_path / TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        print("Tasks download complete.")


if __name__ == "__main__":
    download_dabstep_data()

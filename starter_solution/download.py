from dataclasses import dataclass
from pathlib import Path
import simple_parsing
from huggingface_hub import snapshot_download

@dataclass
class ScriptArgs:
    """Download the dataset per year. Example usage:
    python download.py --year 2023 --dataset_folder dataset
    """
    year: int = 2023 # year to download
    dataset_folder: Path = Path("dataset") # folder to save the dataset

if __name__ == "__main__":
    args = simple_parsing.parse(ScriptArgs)
    snapshot_download(repo_id="hackercupai/hackercup", 
                    repo_type="dataset", local_dir=args.dataset_folder,
                    allow_patterns=[f"{args.year}/*"],
                    force_download=True)
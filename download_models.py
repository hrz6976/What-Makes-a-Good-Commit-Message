import argparse
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve


CHECKPOINT_URLS = {
    "why_bert_bilstm_reproduced.pth": "https://github.com/hrz6976/What-Makes-a-Good-Commit-Message/releases/download/replication-models-v1/why_bert_bilstm_reproduced.pth",
    "what_bert_bilstm_reproduced.pth": "https://github.com/hrz6976/What-Makes-a-Good-Commit-Message/releases/download/replication-models-v1/what_bert_bilstm_reproduced.pth",
}
ALLENNLP_ARCHIVE_URL = "https://zenodo.org/records/6383502/files/elmo-constituency-parser-2018.03.14.tar.gz?download=1"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {destination}")
    urlretrieve(url, destination)


def ensure_checkpoints(checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in CHECKPOINT_URLS.items():
        destination = checkpoint_dir / filename
        if destination.exists():
            print(f"skip existing {destination}")
            continue
        download_file(url, destination)


def ensure_allennlp_model(model_dir: Path) -> None:
    marker = model_dir / "config.json"
    if marker.exists():
        print(f"skip existing {model_dir}")
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "elmo-constituency-parser-2018.03.14.tar.gz"
        download_file(ALLENNLP_ARCHIVE_URL, archive_path)
        print(f"extracting {archive_path} -> {model_dir}")
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pretrained inference checkpoints and preprocessing models.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where the Why/What checkpoints should be stored.",
    )
    parser.add_argument(
        "--allennlp-dir",
        type=Path,
        default=Path("Model"),
        help="Directory where the AllenNLP constituency parser should be extracted.",
    )
    args = parser.parse_args()

    ensure_checkpoints(args.checkpoint_dir)
    ensure_allennlp_model(args.allennlp_dir)


if __name__ == "__main__":
    main()

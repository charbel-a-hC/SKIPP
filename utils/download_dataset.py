import argparse
import os
import zipfile

from huggingface_hub import hf_hub_download


def download_dataset(repo_id, data="train", cache_dir=None):
    print(f"Downloading dataset archive from {repo_id}...")
    archive_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{data}.zip",
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    # Create extraction directory
    extract_dir = os.path.join(
        os.path.dirname(archive_path), f"extracted_{data}"
    )
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        print(f"Extracting archive to {extract_dir}...")
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(path=extract_dir)
    else:
        print(f"Using previously extracted dataset at {extract_dir}")

    return extract_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract dataset from Hugging Face Hub"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Data type to download (train or test)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="expert_data",
        help="Directory to cache the downloaded files",
    )

    args = parser.parse_args()

    repo_id = "charbel-a-h/SKIPP-Expert-Data"

    extract_dir = download_dataset(
        repo_id=repo_id, data=args.data_type, cache_dir=args.cache_dir
    )

    print(f"Dataset downloaded and extracted to: {extract_dir}")


if __name__ == "__main__":
    main()

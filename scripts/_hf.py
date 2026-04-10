"""
HuggingFace Hub upload/download helpers shared by model runners.

Imported by scripts/run_*.py via plain `import _hf` (the runner's dir is
on sys.path automatically when you launch it). Keeps the HF logic out
of each runner's main file without introducing a packaged dependency.
"""

from pathlib import Path


def try_download(repo_id: str, cache_dir: Path, filenames: list[str]) -> bool:
    """Pull the given files from HF into cache_dir. False if the repo or
    any file is missing."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name in filenames:
            src = hf_hub_download(repo_id=repo_id, filename=name)
            Path(src).replace(cache_dir / name)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return False
    print(f"==> pulled {filenames} from hf:{repo_id}")
    return True


def try_upload(repo_id: str, cache_dir: Path, filenames: list[str]) -> None:
    """Push the given files from cache_dir to HF, creating the repo if
    it doesn't exist."""
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, exist_ok=True, repo_type="model")
    api = HfApi()
    for name in filenames:
        api.upload_file(
            path_or_fileobj=str(cache_dir / name),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="model",
        )
    print(f"==> uploaded {filenames} to hf:{repo_id}")

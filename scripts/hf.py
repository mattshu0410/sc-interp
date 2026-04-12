"""HuggingFace Hub upload/download and model-card helpers for runners.

Model repos  (repo_type="model"):  try_download / try_upload
Dataset repos (repo_type="dataset"): try_download_dataset / try_upload_dataset
"""

from pathlib import Path
from typing import Any


def try_download(repo_id: str, cache_dir: Path, filenames: list[str]) -> bool:
    """Pull the given files from HF into cache_dir. Returns True on success,
    False on a missing repo or a missing file, and prints a message
    describing which failure mode fired."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name in filenames:
            src = hf_hub_download(repo_id=repo_id, filename=name)
            Path(src).replace(cache_dir / name)
    except RepositoryNotFoundError:
        print(
            f"==> hf:{repo_id} does not exist yet "
            f"(will be created on next upload if training runs)"
        )
        return False
    except EntryNotFoundError:
        print(
            f"==> hf:{repo_id} exists but is missing one of {filenames} "
            f"(will be re-uploaded after training)"
        )
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


def try_download_dataset(repo_id: str, local_dir: Path, filenames: list[str]) -> bool:
    """Pull the given files from an HF dataset repo into local_dir.

    Returns True on full success, False if the repo or any file is missing.
    Missing-file errors are tolerated individually so partial downloads still
    return whatever was available.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name in filenames:
            try:
                src = hf_hub_download(repo_id=repo_id, filename=name, repo_type="dataset")
                Path(src).replace(local_dir / name)
            except EntryNotFoundError:
                print(f"==> hf dataset:{repo_id} missing {name}, skipping")
                return False
    except RepositoryNotFoundError:
        print(f"==> hf dataset:{repo_id} does not exist yet")
        return False
    print(f"==> pulled {filenames} from hf dataset:{repo_id}")
    return True


def try_upload_dataset(repo_id: str, local_dir: Path, filenames: list[str]) -> None:
    """Push the given files from local_dir to an HF dataset repo.

    Creates the repo if it doesn't exist. Files are uploaded sequentially so
    a partial failure still leaves already-uploaded files intact on HF.
    """
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, exist_ok=True, repo_type="dataset")
    api = HfApi()
    for name in filenames:
        api.upload_file(
            path_or_fileobj=str(local_dir / name),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="dataset",
        )
    print(f"==> uploaded {filenames} to hf dataset:{repo_id}")


def build_model_card(
    *,
    model_display_name: str,
    base_model_desc: str,
    dataset_desc: str,
    github_repo: str,
    github_commit: str,
    runner_path: str,
    manifest_path: str,
    tags: list[str],
    training_summary: str,
    budget_table: dict[str, Any],
    metrics_table: dict[str, dict[str, float]] | None = None,
    paper_comparison: str | None = None,
    known_limitations: list[str] | None = None,
    files: dict[str, str] | None = None,
    usage_snippet: str | None = None,
    citation: str | None = None,
) -> str:
    """Generate a HuggingFace model card README.md for a fine-tuned model.

    Fields are filled in by each runner at upload time. The layout mirrors
    the template used across sc-interp uploads.
    """
    def _table(d: dict[str, Any]) -> str:
        lines = ["| | |", "|---|---|"]
        for k, v in d.items():
            lines.append(f"| {k} | {v} |")
        return "\n".join(lines)

    def _metrics(m: dict[str, dict[str, float]]) -> str:
        if not m:
            return ""
        header = "| metric | mean | median | max |\n|---|---|---|---|"
        rows = [header]
        for name, stats in m.items():
            rows.append(
                f"| {name} | {stats.get('mean', ''):.4f} | "
                f"{stats.get('median', ''):.4f} | {stats.get('max', ''):.4f} |"
            )
        return "\n".join(rows)

    tag_yaml = "\n".join(f"- {t}" for t in tags)
    frontmatter = (
        "---\n"
        "library_name: pytorch\n"
        "pipeline_tag: other\n"
        f"tags:\n{tag_yaml}\n"
        f"datasets:\n- {dataset_desc}\n"
        "---\n"
    )

    body_parts = [
        f"# {model_display_name}\n",
        f"Produced as part of the [sc-interp](https://github.com/{github_repo}) "
        "single-cell model comparison repo.\n",
        "## Provenance\n",
        f"- Source code commit: [`{github_commit[:7]}`]"
        f"(https://github.com/{github_repo}/tree/{github_commit})",
        f"- Runner: [`{runner_path}`](https://github.com/{github_repo}/blob/{github_commit}/{runner_path})",
        f"- Dataset manifest: [`{manifest_path}`](https://github.com/{github_repo}/blob/{github_commit}/{manifest_path})\n",
        "## Base model\n",
        base_model_desc + "\n",
        "## Training\n",
        training_summary,
        "\n### Budget and stopping\n",
        _table(budget_table),
        "\n",
    ]

    if metrics_table:
        body_parts.append("## Test set metrics (cell-eval)\n")
        body_parts.append(_metrics(metrics_table))
        body_parts.append("")
        if paper_comparison:
            body_parts.append(paper_comparison + "\n")

    if known_limitations:
        body_parts.append("## Known limitations\n")
        body_parts.extend(f"- {lim}" for lim in known_limitations)
        body_parts.append("")

    if files:
        body_parts.append("## Files\n")
        for name, desc in files.items():
            body_parts.append(f"- `{name}` — {desc}")
        body_parts.append("")

    if usage_snippet:
        body_parts.append("## Usage\n")
        body_parts.append("```python")
        body_parts.append(usage_snippet)
        body_parts.append("```\n")

    if citation:
        body_parts.append("## Citation\n")
        body_parts.append(citation)

    return frontmatter + "\n".join(body_parts)

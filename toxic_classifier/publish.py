"""העלאת גרסת מודל מהרג'יסטרי ל-Hugging Face Hub."""

from dataclasses import dataclass
from typing import Optional

from huggingface_hub import CommitOperationAdd, HfApi

from .config import BASE_MODEL
from .registry import get_model_path

# קבצים שלא צריך להעלות (נוצרים מקומית בלבד)
_SKIP_FILES = {".DS_Store", "training_args.bin"}


@dataclass
class PublishResult:
    repo_id: str
    revision: str
    url: str


def model_card(repo_id: str, version: str, metrics: Optional[dict] = None) -> str:
    metrics = metrics or {}
    lines = [
        "---",
        "language: he",
        "library_name: transformers",
        "pipeline_tag: text-classification",
        f"base_model: {BASE_MODEL}",
        "tags: [toxicity, hebrew, text-classification]",
        "---",
        "",
        "# Hebrew Toxicity Classifier",
        "",
        f"Fine-tuned [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) that classifies Hebrew comments as toxic or safe.",
        "",
        f"**Version:** `{version}`",
    ]
    if metrics.get("accuracy") is not None:
        lines.append(f"**Accuracy:** {metrics['accuracy']:.2%}  ")
    if metrics.get("f1") is not None:
        lines.append(f"**F1 (weighted):** {metrics['f1']:.2%}  ")
    if metrics.get("dataset_size"):
        lines.append(f"**Training examples:** {metrics['dataset_size']}")
    lines += [
        "",
        "## Usage",
        "",
        "```python",
        "from transformers import pipeline",
        "",
        f'classifier = pipeline("text-classification", model="{repo_id}", revision="{version}")',
        'classifier("איזה יום יפה היום")',
        "```",
        "",
        "Label 1 (`toxic`) is toxic, label 0 (`safe`) is safe. "
        "Models trained before label names were added return `LABEL_1` / `LABEL_0`.",
        "",
    ]
    return "\n".join(lines)


def publish_model(
    version: str,
    repo_id: str,
    private: bool = True,
    metrics: Optional[dict] = None,
    token: Optional[str] = None,
) -> PublishResult:
    """מעלה את הגרסה ל-repo (יוצר אותו אם צריך) ומתייג את הקומיט בשם הגרסה.

    ה-branch הראשי תמיד מכיל את הגרסה האחרונה שהועלתה; גרסאות קודמות זמינות דרך revision=<version>.
    """
    path = get_model_path(version)
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)

    operations = [
        CommitOperationAdd(path_in_repo=f.name, path_or_fileobj=str(f))
        for f in sorted(path.iterdir())
        if f.is_file() and f.name not in _SKIP_FILES
    ]
    operations.append(
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=model_card(repo_id, version, metrics).encode())
    )
    api.create_commit(repo_id, operations=operations, commit_message=f"Upload model {version}")
    api.create_tag(repo_id, tag=version, exist_ok=True)

    return PublishResult(repo_id=repo_id, revision=version, url=f"https://huggingface.co/{repo_id}/tree/{version}")


def default_repo_id(name: str = "hebrew-toxicity", token: Optional[str] = None) -> str:
    """<משתמש-HF>/hebrew-toxicity. זורק שגיאה אם אין התחברות ל-HF."""
    return f"{HfApi(token=token).whoami()['name']}/{name}"

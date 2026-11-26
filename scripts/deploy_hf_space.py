"""Deploy the API container to a Hugging Face Space (Docker SDK).

Run by GitHub Actions on every push to main, or by hand the first time:
    set HF_TOKEN=hf_xxx            (PowerShell: $env:HF_TOKEN="hf_xxx")
    set HF_SPACE_ID=<your-hf-username>/abc-bank-drift-api
    python scripts/deploy_hf_space.py
The Space is created if it does not exist. Only the API code and model files are uploaded:
no data, no MLflow database, no secrets.
"""
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def stage(target: Path) -> None:
    shutil.copy(ROOT / "deploy" / "hf_space" / "README.md", target / "README.md")
    shutil.copy(ROOT / "Dockerfile.api", target / "Dockerfile")
    shutil.copy(ROOT / "requirements-api.txt", target / "requirements-api.txt")
    shutil.copytree(ROOT / "api", target / "api", ignore=shutil.ignore_patterns("__pycache__"))
    (target / "src").mkdir()
    for f in ["__init__.py", "config.py"]:
        shutil.copy(ROOT / "src" / f, target / "src" / f)
    shutil.copytree(ROOT / "src" / "model", target / "src" / "model", ignore=shutil.ignore_patterns("__pycache__"))
    (target / "models").mkdir()
    for f in (ROOT / "models").iterdir():
        if f.suffix in (".joblib", ".json"):
            shutil.copy(f, target / "models" / f.name)


def main():
    token, space = os.environ["HF_TOKEN"], os.environ["HF_SPACE_ID"]
    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space",
                          commit_message="Deploy API from GitHub", delete_patterns=["*"])
    print(f"Deployed: https://huggingface.co/spaces/{space}")
    print(f"API:      https://{space.replace('/', '-').replace('_', '-').lower()}.hf.space/docs")


if __name__ == "__main__":
    main()

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts" / "latest"
REGISTRY_DIR = ROOT / "model_registry"
METADATA_FILE = REGISTRY_DIR / "registry.json"
PROD_LINK = REGISTRY_DIR / "production"

REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def read_metadata():
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text())
    return {"versions": []}

def write_metadata(meta: dict):
    METADATA_FILE.write_text(json.dumps(meta, indent=2))

def next_version_name(meta: dict) -> str:
    # v0001, v0002, ...
    existing = [v["version"] for v in meta["versions"]]
    if not existing:
        return "v0001"
    last = sorted(existing)[-1]
    n = int(last[1:]) + 1
    return f"v{n:04d}"

def latest_version_name(meta: dict) -> str | None:
    if not meta["versions"]:
        return None
    return sorted([v["version"] for v in meta["versions"]])[-1]

def copy_artifact_to_version(version_dir: Path):
    version_dir.mkdir(parents=True, exist_ok=True)
    # Expect artifacts/latest/model.tar.gz + model_card.json
    for fname in ["model.tar.gz", "model_card.json"]:
        src = ARTIFACTS_DIR / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing artifact: {src}")
        shutil.copy2(src, version_dir / fname)

def point_prod_to_version(version_dir: Path):
    # PROD_LINK → model_registry/v000x
    if PROD_LINK.is_symlink() or PROD_LINK.exists():
        PROD_LINK.unlink()
    PROD_LINK.symlink_to(version_dir.name, target_is_directory=True)

@dataclass
class VersionInfo:
    version: str
    created_at: str
    stage: str = "None"  # None | Staging | Production
    metrics: dict | None = None
    notes: str | None = None
import json
from pathlib import Path
from common import (
    REGISTRY_DIR, METADATA_FILE, read_metadata, write_metadata,
    next_version_name, copy_artifact_to_version, VersionInfo, now_iso
)

def main():
    meta = read_metadata()
    version = next_version_name(meta)
    version_dir = REGISTRY_DIR / version
    copy_artifact_to_version(version_dir)

    # Load metrics from model_card.json
    card = json.loads((version_dir / "model_card.json").read_text())
    metrics = card.get("metrics", {})

    vi = VersionInfo(version=version, created_at=now_iso(), stage="Staging", metrics=metrics)
    meta["versions"].append(vi.__dict__)
    write_metadata(meta)

    print(f"Registered {version} (stage=Staging)")
    print(f"Registry: {METADATA_FILE}")

if __name__ == "__main__":
    main()
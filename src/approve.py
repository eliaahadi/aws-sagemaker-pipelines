import argparse
import json
from pathlib import Path
from common import (
    REGISTRY_DIR, PROD_LINK, read_metadata, write_metadata,
    latest_version_name, point_prod_to_version
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="latest", help="version id like v0003 or 'latest'")
    args = parser.parse_args()

    meta = read_metadata()
    if not meta["versions"]:
        raise SystemExit("No versions in registry. Run `make register` first.")

    ver = args.version
    if ver == "latest":
        ver = latest_version_name(meta)
    assert ver is not None

    # Update stages: set chosen ver → Production; demote others
    for v in meta["versions"]:
        v["stage"] = "Production" if v["version"] == ver else "Archived"

    write_metadata(meta)

    # Point production symlink to chosen version dir
    version_dir = REGISTRY_DIR / ver
    point_prod_to_version(version_dir)

    print(f"Promoted {ver} to Production")
    print(f"Production link -> {version_dir}")

if __name__ == "__main__":
    main()
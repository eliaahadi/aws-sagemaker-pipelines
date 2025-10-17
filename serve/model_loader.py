import tarfile
from pathlib import Path
import importlib.util
import tempfile

from ..src.common import REGISTRY_DIR, PROD_LINK

# Because FastAPI runs from ./serve, adjust relative import if needed
# Fallback if relative import fails:
try:
    from src.common import REGISTRY_DIR, PROD_LINK  # type: ignore
except Exception:
    pass

def load_production_model():
    prod_path = (REGISTRY_DIR / PROD_LINK)
    if not prod_path.exists():
        raise RuntimeError("No Production model set. Run `make approve`.")

    version_dir = (REGISTRY_DIR / prod_path)
    tar_path = version_dir / "model.tar.gz"
    if not tar_path.exists():
        raise RuntimeError(f"Missing model.tar.gz in {version_dir}")

    tmpdir = tempfile.mkdtemp(prefix="model_")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(tmpdir)

    # Load model via inference.py contract
    inf_py = Path(tmpdir) / "inference.py"
    spec = importlib.util.spec_from_file_location("inference", str(inf_py))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore

    model = mod.model_fn(tmpdir)  # type: ignore
    return model, mod
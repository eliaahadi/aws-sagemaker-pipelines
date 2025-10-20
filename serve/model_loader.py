import tarfile
from pathlib import Path
import importlib.util
import tempfile

# Prefer an absolute import because `uvicorn serve.app:app` treats `serve` as
# a top-level module. Fall back to the relative import when run as a package.
try:
    # When serve/ is a sibling of src/, this absolute import works.
    from src.common import REGISTRY_DIR, PROD_LINK  # type: ignore
except Exception:
    # When the package is executed as a package (e.g. python -m), the
    # relative import may be necessary.
    from ..src.common import REGISTRY_DIR, PROD_LINK
except Exception:
    # As a last resort (different import contexts), derive the registry
    # paths from this file's location. This avoids ImportError when the
    # module is imported via different entrypoints (uvicorn reloader/spawn).
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    # repo_root should be the project root; src/ is a sibling of serve/
    src_root = repo_root / "src"
    REGISTRY_DIR = src_root / ".." / "model_registry"
    PROD_LINK = REGISTRY_DIR / "production"

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
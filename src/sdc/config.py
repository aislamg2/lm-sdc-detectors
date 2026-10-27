"""Load configs/default.yaml (optionally overlaid with another YAML file)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Return the experiment config as a nested dict.

    TODO: read DEFAULT_CONFIG with yaml.safe_load, deep-merge `path` on top,
    and resolve every entry under `paths` against REPO_ROOT.
    """
    raise NotImplementedError

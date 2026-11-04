"""Load configs/default.yaml (optionally overlaid with another YAML file)."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _merge(base: dict, overlay: dict, where: str = "") -> dict:
    """Return `base` with `overlay` on top; unknown keys are errors, so typos can't pass silently."""
    merged = dict(base)
    for key, value in overlay.items():
        name = f"{where}{key}"
        if key not in base:
            raise ValueError(f"unknown config key: {name}")
        if isinstance(base[key], dict) != isinstance(value, dict):
            raise ValueError(f"config key {name} must keep its shape (section vs. value)")
        merged[key] = _merge(base[key], value, f"{name}.") if isinstance(value, dict) else value
    return merged


def load_config(path: str | Path | None = None) -> dict:
    """Return the experiment config as a nested dict.

    Reads DEFAULT_CONFIG, deep-merges `path` on top, and resolves every entry
    under `paths` against REPO_ROOT (absolute paths are left as they are).
    """
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if path is not None:
        overlay = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(overlay, dict):
            raise ValueError(f"config file {path} must contain a mapping")
        config = _merge(config, overlay)
    config["paths"] = {name: str(REPO_ROOT / value) for name, value in config["paths"].items()}
    return config

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(name: str, profile: str | None = None) -> dict:
    """Load configs/<name>.yaml relative to the project root. `profile` (e.g. "india") overlays the
    values under `profiles: <profile>:` in that file onto the defaults."""
    path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    profiles = cfg.pop("profiles", {}) or {}
    if profile is not None:
        if profile not in profiles:
            raise KeyError(f"No profile '{profile}' in {path.name} (have: {sorted(profiles)})")
        for section, values in profiles[profile].items():
            cfg.setdefault(section, {}).update(values)
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a path from a config relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p

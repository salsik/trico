from __future__ import annotations

import os
from pathlib import Path

def data_dir() -> Path:
    """
    Returns the root directory for data files.

    Priority:
      1) $UIFLOW_DATA_DIR
      2) <repo_root>/data   (best-effort fallback)
    """
    # 1) explicit override
    env = os.getenv("UIFLOW_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()

    # 2) fallback: try to locate repo root by walking up from this file
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return (p / "data").resolve()

    # last resort: current working directory / data
    return (Path.cwd() / "data").resolve()


def data_path(*parts: str) -> Path:
    """Convenience: data_path("processed", "clip_meta.tsv")"""
    return data_dir().joinpath(*parts)

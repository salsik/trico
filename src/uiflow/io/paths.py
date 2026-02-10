import os
from dataclasses import dataclass

@dataclass(frozen=True)
class ScreenRef:
    """Canonical identifier for a screen."""
    app_id: str
    trace_id: str
    screen_id: str

    @property
    def screen_key(self) -> str:
        return f"{self.app_id}::{self.trace_id}::{self.screen_id}"

def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def parse_from_view_hierarchy_path(path: str) -> ScreenRef:
    """
    Expected:
      root/app_id/trace_id/view_hierarchies/<screen_id>.json
    """
    parts = os.path.normpath(path).split(os.sep)
    if "view_hierarchies" not in parts:
        raise ValueError(f"Not a view_hierarchies path: {path}")
    idx = len(parts) - 1 - parts[::-1].index("view_hierarchies")  # last occurrence
    if idx < 2:
        raise ValueError(f"Path too short to parse app/trace: {path}")
    app_id = parts[idx - 2]
    trace_id = parts[idx - 1]
    screen_id = stem(path)
    return ScreenRef(app_id, trace_id, screen_id)

def parse_from_screenshot_path(path: str) -> ScreenRef:
    """
    Expected:
      root/app_id/trace_id/screenshots/<screen_id>.(png/jpg/...)
    """
    parts = os.path.normpath(path).split(os.sep)
    if "screenshots" not in parts:
        raise ValueError(f"Not a screenshots path: {path}")
    idx = len(parts) - 1 - parts[::-1].index("screenshots")
    if idx < 2:
        raise ValueError(f"Path too short to parse app/trace: {path}")
    app_id = parts[idx - 2]
    trace_id = parts[idx - 1]
    screen_id = stem(path)
    return ScreenRef(app_id, trace_id, screen_id)

def make_screen_key(app_id: str, trace_id: str, screen_id: str) -> str:
    return f"{app_id}::{trace_id}::{screen_id}"

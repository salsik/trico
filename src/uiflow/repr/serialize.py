from __future__ import annotations
from typing import Any, Dict, List, Tuple

TYPE_KEYS = ["class", "className", "type", "resource-id", "resource_id"]
TEXT_KEYS = ["text", "content-desc", "contentDescription", "hint", "label", "accessibilityText"]

def _collect_nodes(obj: Any) -> List[Dict]:
    nodes: List[Dict] = []
    if isinstance(obj, dict):
        nodes.append(obj)
        for v in obj.values():
            nodes.extend(_collect_nodes(v))
    elif isinstance(obj, list):
        for it in obj:
            nodes.extend(_collect_nodes(it))
    return nodes

def extract_pairs(view_json: Any) -> List[Tuple[str, str]]:
    """
    Extract (ui_type, best_text) pairs from a view hierarchy object.
    Works best for RICO-like JSON but degrades gracefully.
    """
    nodes = _collect_nodes(view_json)
    pairs: List[Tuple[str, str]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue

        # visibility filters (best-effort)
        if n.get("visible") is False or n.get("isVisibleToUser") is False:
            continue

        ui_type = None
        for k in TYPE_KEYS:
            v = n.get(k)
            if isinstance(v, str) and v.strip():
                ui_type = v.strip()
                break
        if ui_type is None:
            continue

        best_text = ""
        for k in TEXT_KEYS:
            v = n.get(k)
            if isinstance(v, str) and v.strip():
                best_text = " ".join(v.strip().split())
                break

        pairs.append((ui_type, best_text))
    return pairs

def serialize_screen(
    pairs: List[Tuple[str, str]],
    max_elems: int = 200,
) -> str:
    """
    Produce compact, normalized text for text-embedding / SBERT.
    Dedupes (type,text) while preserving order; truncates length by element count.
    """
    seen = set()
    lines: List[str] = ["UI_SCREEN"]
    for t, s in pairs:
        t2 = t.split(".")[-1].strip()
        key = (t2, s)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{t2}: {s}" if s else f"{t2}")
        if len(lines) - 1 >= max_elems:
            break
    return "\n".join(lines)

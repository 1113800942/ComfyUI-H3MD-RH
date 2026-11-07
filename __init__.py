"""H3MD RH - the RunningHub build of H3MD Director (ComfyUI-H3MD-Director).

Registers only the RH nodes (class names end in RH).  The shared engine in h3md_rh_core/ is a private copy synced
from the main repo; it registers no nodes and no HTTP routes, so this pack can sit next to any version of
ComfyUI-H3MD-Director.  No web extension.
"""
import logging

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from . import h3md_rh_nodes as _nodes
    NODE_CLASS_MAPPINGS.update(_nodes.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_nodes.NODE_DISPLAY_NAME_MAPPINGS)
except Exception as _exc:
    logging.warning("[H3MD RH] nodes not loaded: %s", _exc)


def _version():
    import os
    import re
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyproject.toml"), encoding="utf-8") as f:
            m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
        return m.group(1) if m else "?"
    except Exception:
        return "?"


H3MD_RH_VERSION = _version()
if "H3MDDirectorRH" in NODE_DISPLAY_NAME_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["H3MDDirectorRH"] += f" v{H3MD_RH_VERSION}"
logging.info("[H3MD RH] v%s loaded: %s", H3MD_RH_VERSION, ", ".join(NODE_CLASS_MAPPINGS))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

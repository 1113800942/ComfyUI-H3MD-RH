"""Sync the shared H3MD engine from the main repo (ComfyUI-H3MD-Director) into h3md_rh_core/.

Usage:  python tools/sync_from_main.py <path to main repo checkout>

The core modules are copied unchanged except for one thing: no HTTP routes are registered at import time.
On RunningHub the main pack (0.1.x) may be installed next to this one, and duplicate /h3md/... routes must not
clash.  Only the RH nodes (h3md_rh_*.py at the repo root) are registered with ComfyUI; the core modules are a
private library.  The main repo commit is written to h3md_rh_core/SYNC_FROM.txt.
"""
import os
import re
import shutil
import subprocess
import sys

CORE = [
    "h3md_autosplit.py", "h3md_cache.py", "h3md_depth_control.py", "h3md_director.py", "h3md_director_refmod.py",
    "h3md_drift_correct.py", "h3md_face_refine_save.py", "h3md_ffchunk.py", "h3md_loop.py", "h3md_loop_split.py",
    "h3md_lora.py", "h3md_lowres_anchor.py", "h3md_paths.py", "h3md_refwindow.py", "h3md_sampler.py",
    "h3md_seam.py", "h3md_semantic_bridge.py", "h3md_stream_save.py", "h3md_tiled_hires.py", "h3md_vram.py",
    "h3md_control_panel.py", "h3md_video_preview.py",
]
CORE_DIRS = ["h3md_luna"]
# import-time route registration -> disabled in the RH build
ROUTE_CALL = re.compile(r"^(\s*)register_routes\(\)\s*$", re.M)


def main(src):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dst = os.path.join(here, "h3md_rh_core")
    os.makedirs(dst, exist_ok=True)
    for name in CORE:
        with open(os.path.join(src, name), encoding="utf-8") as f:
            text = f.read()
        text, n = ROUTE_CALL.subn(r"\1pass  # RH build: no HTTP routes (sync_from_main.py)", text)
        if n:
            print(f"  {name}: {n} route registration(s) disabled")
        with open(os.path.join(dst, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    for d in CORE_DIRS:
        target = os.path.join(dst, d)
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(os.path.join(src, d), target, ignore=shutil.ignore_patterns("__pycache__"))
    with open(os.path.join(dst, "__init__.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write('"""Shared H3MD engine, synced from ComfyUI-H3MD-Director by tools/sync_from_main.py. Do not edit here."""\n')
    try:
        commit = subprocess.check_output(["git", "-C", src, "log", "-1", "--format=%H %s"], text=True).strip()
    except Exception:
        commit = "unknown"
    with open(os.path.join(src, "pyproject.toml"), encoding="utf-8") as f:
        version = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M).group(1)
    with open(os.path.join(dst, "SYNC_FROM.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"main version {version}\nmain commit {commit}\n")
    print(f"synced {len(CORE)} modules + {CORE_DIRS} from main {version} ({commit[:12]})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "comfy-h3md-director"))

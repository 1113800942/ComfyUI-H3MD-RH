# H3MD RH

RunningHub build of [H3MD Director](https://github.com/1113800942/comfy-h3md-director) (MiniMax H3 long video).
For use on RunningHub only; on your own PC install ComfyUI-H3MD-Director instead.

| Node | What it does |
|---|---|
| H3MD Director RH | Template (mv / motion_copy / story / custom) + photo, song, guide video from ComfyUI's own Load Image / Load Audio / Load Video -> H3MD plan |
| H3MD Loop Sampler RH | Generates the segments (cache in ComfyUI temp) |
| H3MD Save Video RH | Joins and saves the video (optional face refine / interpolation) |
| H3MD TARA RH | TARA temporal alignment for motion copy (from ComfyUI-BFSNodes by Alisson Anjos, GPL-3.0) |

Class names end in RH, so it installs next to any version of ComfyUI-H3MD-Director and ComfyUI-BFSNodes.
No extra pip packages. Licence: AGPL-3.0-or-later; TARA files GPL-3.0 — see NOTICE.md.

Maintainers: `python tools/sync_from_main.py <main repo>` copies the shared engine after every main release.

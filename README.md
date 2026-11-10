# H3MD RH

RunningHub build of [H3MD Director](https://github.com/1113800942/comfy-h3md-director) (MiniMax H3 long video).
For use on RunningHub only; on your own PC install ComfyUI-H3MD-Director instead.

**For RunningHub staff: [RUNNINGHUB.md](RUNNINGHUB.md)** lists what the pack does on your servers (cache in temp,
8-bit, deleted after saving; reads / writes; no network, no routes; limits; checks before a run), point by point.

| Node | What it does |
|---|---|
| H3MD Director RH | Template (mv / motion_copy / story / custom) + photo, song, guide video from ComfyUI's own Load Image / Load Audio / Load Video -> H3MD plan. At most 30 s and 1 MP. `seamless` (on): motion copy as one seamless take (CCW), segment by segment when off or when the core MiniMax H3 context windows are missing. Motion copy needs a guide that is one continuous shot (no cuts) and follows its camera; empty `scene` keeps the guide's own surroundings (any guide: dance in place, running, tracking), a scene replaces them (the summary warns when the guide's camera moves) |
| H3MD Loop Sampler RH | Generates the video. 8-bit cache in ComfyUI temp, leftovers of an earlier task removed first, temp disk checked before the run |
| H3MD Save Video RH | Joins and saves the video (optional face refine / interpolation), deletes the cache after saving |
| H3MD TARA RH | TARA temporal alignment for motion copy (from ComfyUI-BFSNodes by Alisson Anjos, GPL-3.0) |

Class names end in RH, so it installs next to any version of ComfyUI-H3MD-Director and ComfyUI-BFSNodes.
No extra pip packages. Licence: AGPL-3.0-or-later; TARA files GPL-3.0 — see NOTICE.md.

Example workflows: `workflows/` (MV from a photo and a song, motion copy, multi-shot story).

Maintainers: `python tools/sync_from_main.py <main repo>` copies the shared engine after every main release;
`python tests/test_rh_load.py` checks the RH nodes; `python tools/make_rh_workflows.py <object_info.json> workflows`
rebuilds the example workflows.

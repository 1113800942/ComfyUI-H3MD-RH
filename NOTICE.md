# NOTICE — H3MD RH (ComfyUI-H3MD-RH)

Copyright (C) 2026 YJ. SPDX-License-Identifier: AGPL-3.0-or-later (same terms and attribution term as
ComfyUI-H3MD-Director, see its NOTICE.md: <https://github.com/1113800942/comfy-h3md-director>).

This is the RunningHub build of H3MD Director. `h3md_rh_core/` is an unmodified copy of the main repo's engine
(see `h3md_rh_core/SYNC_FROM.txt`), except that it registers no HTTP routes.  `h3md_rh_core/h3md_luna/`: GPL-3.0,
see its THIRD_PARTY_NOTICES.md.

## Third-party code: TARA (h3md_rh_tara/)

`h3md_rh_tara/` is vendored from **ComfyUI-BFSNodes** by **Alisson Anjos**
(<https://github.com/alisson-anjos/ComfyUI-BFSNodes>), commit 1ebfd0c9b6d40b43ca6c12c05143238fedbfc41e
(v1.118.0), licensed **GPL-3.0** (full text: `h3md_rh_tara/LICENSE-GPL-3.0.txt`).  TARA = native temporal
alignment of MiniMax H3 references for the motion-transfer LoRA.  GPL-3.0 section 13 permits combining it with
this AGPL-3.0 work; these files stay under GPL-3.0.

Files copied unchanged: minimax_h3_native_temporal.py, minimax_h3_native_temporal_core.py,
minimax_h3_temporal_fast.py, minimax_h3_temporal_kitchen.py, minimax_h3_temporal_sol.py,
minimax_h3_sparse_compat.py, minimax_h3_reference_binding.py, minimax_h3_token_weights.py, TARA_PARAMETERS.md.

Modified (2026-11-06, YJ): minimax_h3_temporal_kernel.py — the first-use download of prebuilt binaries
(minimax_h3_temporal_bootstrap.py) was removed; this build never downloads code.  The optional Kitchen one-pass
kernel is used only if `comfy_kitchen.temporal_attention` is already installed.

The node `H3MD TARA RH` is a renamed subclass of `BFSMiniMaxH3NativeTemporal` (same inputs, same behaviour), so it
does not clash with ComfyUI-BFSNodes installed next to it.  All credit for TARA goes to its author.

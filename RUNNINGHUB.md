# H3MD RH 0.4.3-rh1 — 给 RunningHub 的说明 / Notes for RunningHub

**中文在前，English below.**

这个节点包是 [H3MD Director](https://github.com/1113800942/comfy-h3md-director)（MiniMax H3 长视频）专门为 RunningHub 做的版本，只在 RunningHub 上用。下面逐条说明它在 RH 服务器上会做什么、不会做什么，每一条都写了在代码里哪里、怎么核对。

## 一、一览

| 项目 | 内容 |
|---|---|
| 包名 / 版本 | `ComfyUI-H3MD-RH` 0.4.3-rh1（引擎 = 主仓库 0.4.3，`h3md_rh_core/SYNC_FROM.txt` 记着来源 commit） |
| 节点 | 4 个，类名都以 RH 结尾，和任何版本的 ComfyUI-H3MD-Director、ComfyUI-BFSNodes 都不撞名：`H3MDDirectorRH`、`H3MDLoopSamplerRH`、`H3MDSaveVideoRH`、`H3MDTaraRH` |
| 额外的 pip 包 | **没有**（只用 ComfyUI 自带的 torch、numpy、PyAV、Pillow、transformers） |
| 前端 / 网页接口 | **没有**：不带 `WEB_DIRECTORY`（没有前端脚本），不注册任何 HTTP 路由 |
| 联网 / 下载 | **没有**：运行时不访问网络、不下载模型或程序、不 pip install |
| 许可证 | AGPL-3.0-or-later；`h3md_rh_tara/` 来自 ComfyUI-BFSNodes（GPL-3.0），见 `NOTICE.md`。作者同意 RunningHub 安装和使用本包 |

## 二、按 RH 的规则逐条说明

| # | 规则 | 我们怎么做 | 代码位置 | 怎么核对 |
|---|---|---|---|---|
| 1 | 缓存只放临时目录 | 分段缓存固定在 **ComfyUI temp/`h3md_rh_cache`**。多人共用的 ComfyUI（`--listen`）上，就算工作流里填了别的 `cache_dir`，也只允许 ComfyUI 的 output / temp | `h3md_rh_nodes.py` `H3MDLoopSamplerRH.run`；`h3md_rh_core/h3md_cache.py` `allowed_cache_root` | 开跑日志 `[H3MD RH] machine: … temp disk in …/h3md_rh_cache` |
| 2 | 缓存 8 位 | Loop RH **强制** `cache_dtype = uint8`（工作流里改成 fp16 也会被改回来），每个像素 3 字节，比 fp16 小一半，画面看不出差别（11-08 实测：保存的视频每个像素最多差 1/255） | `H3MDLoopSamplerRH.run`（`CACHE_DTYPE`） | 日志 `cache_dtype fp16 -> uint8`；`tests/test_rh_load.py` |
| 3 | 缓存多大 | 帧数 × 宽 × 高 × 3 字节。本包上限 30 秒、1 MP：最多 719 帧 × 1344×768 × 3 ≈ **2.2 GB** | `cache_estimate_bytes` | 开跑日志里有这一次的估算 |
| 4 | 出片就删 | Save RH 的 `delete_cache_after_save` **默认开**：视频文件完整写完、关闭、改好名以后，删除这一次的缓存文件夹；只删看起来是 H3MD 缓存的文件夹（`run_*` / `resume_*`，里面只有 `seg_*` 和这一次的 `run.json`），有别的文件就不删 | `h3md_rh_core/h3md_stream_save.py` `delete_cache_dir`；`h3md_face_refine_save.py`；`H3MDSaveVideoRH.save` | 日志 `[H3MD RH] video saved; segment cache: deleted after save` |
| 5 | 不留垃圾 | ① 每次开跑前，Loop RH 先清空 `temp/h3md_rh_cache` 里上一次任务留下的东西（上次出错、没走到保存时会留）；② 照片、歌、参考视频的拷贝放在 `temp/h3md_rh_media`，每次 Director RH 运行时先删掉上一次的再写这一次的 | `clear_old_cache`、`clear_media` | 日志 `removed N leftover cache folder(s)`；测试 |
| 6 | 不跨用户复用 | 每次运行一个单独的 `run_<时间>_<进程>` 文件夹。多人共用的 ComfyUI 上，"从缓存续跑" 默认关，不会拿到别人的分段 | `h3md_cache.py` `SegmentCache` | 日志 `shared ComfyUI: resume_from_cache off` |
| 7 | 读文件的范围 | 只读 ComfyUI 的 input / output / temp。用户素材全部从核心 `LoadImage` / `LoadAudio` / `LoadVideo` 接进来（AI 应用能直接开放这些输入），Director RH 把它们存成 temp 里的文件再写进计划 | `h3md_rh_core/h3md_paths.py` `may_read`、`check_read`；`h3md_rh_nodes.py` `save_image` / `save_audio` / `save_video` | 计划里的文件路径都在 temp 里 |
| 8 | 写文件的范围 | 只写 ComfyUI 的 temp（缓存、素材拷贝）和 output（成片，文件名前缀 `H3MD_RH`） | `h3md_paths.py` `may_write`、`output_root` | — |
| 9 | 时长封顶 | 整条视频**最多 30 秒**，三个模板都管：MV 放 3 个 9.4 秒的镜头；故事模板的镜头一起缩短（最短 4.5 秒，放不下的行去掉并在摘要里写明）；动作复刻最多用原片 30 秒。`custom` 模板的计划也查，超了直接报错 | `h3md_rh_templates.py`（`MAX_SECONDS`）；`h3md_rh_nodes.py` `check_limits` | 测试；Director 日志 `… frames (x s)` |
| 10 | 分辨率封顶 | 每帧**最多 1 MP**（16:9 是 1344×768）。输入框最大 1.0，`custom` 计划超过也报错 | `MAX_MEGAPIXELS`、`check_limits` | 测试 |
| 11 | 开跑前检查 | Loop RH 开跑前在日志里打出 ComfyUI 版本、显卡和显存、空闲内存、临时盘剩余，以及这一次估算要多少临时盘；**不够就在编码之前直接报错**（中英文），不会跑到一半爆盘 | `preflight` | 日志 `[H3MD RH] machine: …` |
| 12 | 显存 | **目标：24 GB 显卡（4090）能跑。** 高清阶段按空闲显存自动切条；动作复刻的参考视频统一按短边 480 编码（1080p 参考视频在高清阶段会把 32 GB 卡占满）。11-09 实测：1 MP 动作复刻，把显存限制在约 24 GB（32 GB 卡留出 6 GB），12 秒成片一次跑完，用时 590 秒，高清阶段只切 1 条；MV 第一个镜头 9.4 秒，220 秒 | `h3md_rh_core/h3md_tiled_hires.py`、`h3md_window.py` | Loop 日志里每一遍的显存峰值（peak） |
| 13 | 内存 | 无接缝模式一组的长度按机器内存自动定，解码分块（约 10 秒一块） | `h3md_window.py` `auto_group_length` | Director 摘要 `window mode: groups up to …` |
| 14 | 无接缝开关 / 缺节点时 | 动作复刻的 `seamless`（接续窗口 CCW）**默认关**，按分段式出片：11-09 实测和 TARA 一起用时，两个窗口交接处还可能跳一下，修好之前不默认开。打开时要用 ComfyUI 核心的 `MiniMaxH3ContextWindows`；没有这个节点时 Director RH **自动改成分段式**，摘要里写一句，不报错，照样出片 | `core_windows_available` | 摘要 `rendered segment by segment instead` |
| 15 | 任务时长 | 时长和分辨率都封顶，一次任务就是一条完整视频（工作流 `stop_after = 0`），不需要跨任务续跑 | 工作流 | — |
| 16 | 不碰别人的东西 | 不改 ComfyUI 核心行为、不改别的节点；TARA 是 BFSNodes 节点的改名子类，不影响装在旁边的 BFSNodes | `h3md_rh_nodes.py` `_tara_class` | — |
| 17 | 不执行外部程序 | 没有 subprocess、eval / exec、读环境变量。唯一的动态导入：TARA 内核在已经装了 `comfy_kitchen.temporal_attention` 时才用它 | `h3md_rh_tara/minimax_h3_temporal_kernel.py` | `grep` |

一台 ComfyUI 同一时间只跑一个任务（ComfyUI 正常的队列），第 5 条的"清上一次"是按这个前提做的。

## 三、工作流和模型

`workflows/` 里三张示例（都是一次跑完整条，最长 30 秒）：MV 照片加歌、动作复刻、多镜头故事。第 ① 组是要在 AI 应用里开放的输入。

| 工作流用到的节点 | 来自 |
|---|---|
| `UNETLoader`、`CLIPLoader`（type minimax）、`VAELoader`、`LoraLoaderModelOnly`、`KSamplerSelect`、`BasicScheduler`、`LoadImage`、`LoadAudio`、`LoadVideo`、`MiniMaxH3ContextWindows` | ComfyUI 核心 |
| `MiniMaxChunkFeedForward`、`MiniMaxH3MemoryEfficientSageAttentionPatch` | ComfyUI-KJNodes |
| `H3SigmaRefiner` | ComfyUI-YCNodes-MiniMax-H3 |
| `H3MDDirectorRH`、`H3MDLoopSamplerRH`、`H3MDSaveVideoRH`、`H3MDTaraRH` | 本包 |

模型（文件名是我们本机的，RH 上按 RH 的文件名重新选）：MiniMax H3 hybrid b25-49 int8、Qwen3-VL 32B int8 文本编码器、视频 / 音频 VAE、H3 latent upscaler、turbo 768 加速 LoRA；动作复刻另加 TARA 动作迁移 LoRA（`minimax_h3_motion_transfer_tara_v1.0_ref2va`，RH 模型库里已经有）。

TARA 节点：RH 上没有 ComfyUI-BFSNodes 的 `BFSMiniMaxH3NativeTemporal`，所以它的代码整份带在本包的 `h3md_rh_tara/` 里，节点名 `H3MDTaraRH`，不用另装 BFSNodes，也不用另装 pip 包。

## 四、怎么核对

- 测试：`python tests/test_rh_load.py`（需要 ComfyUI 路径：`COMFYUI_ROOT`）。覆盖：30 秒和 1 MP 上限（含 custom 计划）、无接缝默认关、打开后缺节点时改分段、出片后删缓存（含 `run.json` 的文件夹）、强制 8 位、开跑前清旧缓存、临时盘不够时报错、素材拷贝只留这一次的。
- 跑一次看日志：`[H3MD RH] machine: …`（机器和临时盘）→ `removed … leftover cache`（如果有）→ `template …: … frames (… s), …x…, seamless / segment by segment` → 保存后 `video saved; segment cache: deleted after save`。

## 五、联系

作者 YJ（YouTube: AI 纵横 @yj0942）。主仓库：<https://github.com/1113800942/comfy-h3md-director>。

---

# English

This pack is the RunningHub build of [H3MD Director](https://github.com/1113800942/comfy-h3md-director) (MiniMax H3 long video). It is for RunningHub only. Below: what it does and does not do on RH servers, with the code location and how to check each point.

## 1. At a glance

| Item | Detail |
|---|---|
| Package / version | `ComfyUI-H3MD-RH` 0.4.3-rh1 (engine = main repo 0.4.3; source commit in `h3md_rh_core/SYNC_FROM.txt`) |
| Nodes | 4, all class names end in RH, so no clash with any ComfyUI-H3MD-Director or ComfyUI-BFSNodes: `H3MDDirectorRH`, `H3MDLoopSamplerRH`, `H3MDSaveVideoRH`, `H3MDTaraRH` |
| Extra pip packages | **None** (only what ComfyUI ships: torch, numpy, PyAV, Pillow, transformers) |
| Frontend / web API | **None**: no `WEB_DIRECTORY`, no HTTP routes |
| Network / downloads | **None**: no network access, no model or code downloads, no pip install at run time |
| Licence | AGPL-3.0-or-later; `h3md_rh_tara/` from ComfyUI-BFSNodes (GPL-3.0), see `NOTICE.md`. The author agrees to RunningHub installing and running this pack |

## 2. RunningHub rules, point by point

| # | Rule | What the pack does | Code | How to check |
|---|---|---|---|---|
| 1 | Cache in temp only | The segment cache always lives in **ComfyUI temp/`h3md_rh_cache`**. On a shared ComfyUI (`--listen`) any other `cache_dir` is only accepted inside ComfyUI's output / temp | `h3md_rh_nodes.py` `H3MDLoopSamplerRH.run`; `h3md_rh_core/h3md_cache.py` `allowed_cache_root` | start-of-run log `[H3MD RH] machine: … temp disk in …/h3md_rh_cache` |
| 2 | 8-bit cache | Loop RH **forces** `cache_dtype = uint8` (fp16 in a workflow is switched back): 3 bytes a pixel, half of fp16, no visible difference (measured 11-08: saved video differs by at most 1/255 per pixel) | `H3MDLoopSamplerRH.run` (`CACHE_DTYPE`) | log `cache_dtype fp16 -> uint8`; `tests/test_rh_load.py` |
| 3 | Cache size | frames × width × height × 3 bytes. With this pack's limits (30 s, 1 MP): at most 719 frames × 1344×768 × 3 ≈ **2.2 GB** | `cache_estimate_bytes` | the estimate is in the start-of-run log |
| 4 | Deleted after the video is saved | Save RH has `delete_cache_after_save` **on by default**: once the video file is fully written, closed and renamed, this run's cache folder is deleted; only folders that look like an H3MD cache (`run_*` / `resume_*` holding only `seg_*` and the run's `run.json`) are ever deleted; any other file there and the folder is kept | `h3md_rh_core/h3md_stream_save.py` `delete_cache_dir`; `h3md_face_refine_save.py`; `H3MDSaveVideoRH.save` | log `[H3MD RH] video saved; segment cache: deleted after save` |
| 5 | Nothing left behind | ① Before each run Loop RH empties what an earlier task left in `temp/h3md_rh_cache` (e.g. a run that failed before saving); ② copies of the photo / song / guide video go to `temp/h3md_rh_media`, and Director RH removes the previous task's copies before writing this task's | `clear_old_cache`, `clear_media` | log `removed N leftover cache folder(s)`; tests |
| 6 | No reuse across users | Every run gets its own `run_<time>_<pid>` folder. On a shared ComfyUI, resuming from an old cache is off, so nobody picks up someone else's segments | `h3md_cache.py` `SegmentCache` | log `shared ComfyUI: resume_from_cache off` |
| 7 | Reads | Only ComfyUI's input / output / temp. All user media come in through the core `LoadImage` / `LoadAudio` / `LoadVideo` (so AI apps can expose them); Director RH saves them as files in temp and puts those into the plan | `h3md_rh_core/h3md_paths.py` `may_read`, `check_read`; `h3md_rh_nodes.py` `save_image` / `save_audio` / `save_video` | every media path in the plan is in temp |
| 8 | Writes | Only ComfyUI temp (cache, media copies) and output (the video, prefix `H3MD_RH`) | `h3md_paths.py` `may_write`, `output_root` | — |
| 9 | Length cap | The whole video is **at most 30 s**, in every template: MV makes 3 shots of 9.4 s; story shortens all shots together (at least 4.5 s each; lines that do not fit are dropped and named in the summary); motion copy uses at most 30 s of the guide. A `custom` plan is checked too and refused when longer | `h3md_rh_templates.py` (`MAX_SECONDS`); `h3md_rh_nodes.py` `check_limits` | tests; Director log `… frames (x s)` |
| 10 | Resolution cap | **At most 1 MP** a frame (1344×768 at 16:9). The input stops at 1.0; a larger `custom` plan is refused | `MAX_MEGAPIXELS`, `check_limits` | tests |
| 11 | Check before running | Loop RH logs the ComfyUI version, GPU and VRAM, free RAM, free temp disk and how much temp disk this run needs; **when it will not fit, it stops with a clear message (Chinese + English) before anything is encoded**, instead of filling the disk half way | `preflight` | log `[H3MD RH] machine: …` |
| 12 | VRAM | **Target: runs on a 24 GB card (4090).** The high-res pass is split into strips by the free VRAM; the motion-copy guide video is always encoded at short edge 480 (a 1080p guide filled a 32 GB card in the high-res pass). Measured 11-09: 1 MP motion copy with VRAM held to about 24 GB (6 GB reserved on a 32 GB card) rendered a 12 s video in one go, 590 s, high-res pass in 1 strip; one 9.4 s MV shot took 220 s | `h3md_rh_core/h3md_tiled_hires.py`, `h3md_window.py` | peak VRAM per pass in the Loop log |
| 13 | RAM | In seamless mode the group length follows the machine's RAM; decoding is chunked (~10 s chunks) | `h3md_window.py` `auto_group_length` | Director summary `window mode: groups up to …` |
| 14 | Seamless switch / missing node | Motion copy's `seamless` (CCW, chained context windows) is **off by default**, so it renders segment by segment: in the 11-09 test with TARA a jump could still show where two windows hand over; it stays off until that is fixed. When switched on it builds on ComfyUI's core `MiniMaxH3ContextWindows`; without it, Director RH **falls back to segment-by-segment by itself**, says so in the summary, no error | `core_windows_available` | summary `rendered segment by segment instead` |
| 15 | Task time | Length and resolution are capped, so one task = one whole video (workflows use `stop_after = 0`); no resume across tasks needed | workflows | — |
| 16 | Leaves others alone | Does not change ComfyUI core behaviour or other nodes; TARA is a renamed subclass of the BFSNodes node and does not touch a BFSNodes installed next to it | `h3md_rh_nodes.py` `_tara_class` | — |
| 17 | No external programs | No subprocess, eval / exec or environment variables. The only dynamic import: the TARA kernel uses `comfy_kitchen.temporal_attention` only when it is already installed | `h3md_rh_tara/minimax_h3_temporal_kernel.py` | `grep` |

The cleanup in point 5 assumes one task at a time per ComfyUI server (ComfyUI's normal queue).

## 3. Workflows and models

`workflows/` holds three examples (each renders the whole video, at most 30 s): MV from a photo and a song, motion copy, multi-shot story. Group ① holds the inputs an AI app should expose.

| Node used by the workflows | From |
|---|---|
| `UNETLoader`, `CLIPLoader` (type minimax), `VAELoader`, `LoraLoaderModelOnly`, `KSamplerSelect`, `BasicScheduler`, `LoadImage`, `LoadAudio`, `LoadVideo`, `MiniMaxH3ContextWindows` | ComfyUI core |
| `MiniMaxChunkFeedForward`, `MiniMaxH3MemoryEfficientSageAttentionPatch` | ComfyUI-KJNodes |
| `H3SigmaRefiner` | ComfyUI-YCNodes-MiniMax-H3 |
| `H3MDDirectorRH`, `H3MDLoopSamplerRH`, `H3MDSaveVideoRH`, `H3MDTaraRH` | this pack |

Models (our local file names; pick RH's own names): MiniMax H3 hybrid b25-49 int8, Qwen3-VL 32B int8 text encoder, video / audio VAE, H3 latent upscaler, turbo 768 LoRA; motion copy also uses the TARA motion-transfer LoRA (`minimax_h3_motion_transfer_tara_v1.0_ref2va`, already in RH's model library).

TARA node: RH does not have ComfyUI-BFSNodes' `BFSMiniMaxH3NativeTemporal`, so its code ships whole in this pack's `h3md_rh_tara/` as `H3MDTaraRH`; no BFSNodes install and no pip packages needed.

## 4. How to check

- Tests: `python tests/test_rh_load.py` (ComfyUI path in `COMFYUI_ROOT`). Covers: 30 s and 1 MP limits (custom plans too), seamless off by default and the fallback without the core node when on, deleting the cache after saving (folders holding `run.json` too), forced 8-bit cache, clearing leftovers before a run, the clear error when temp disk is short, only this task's media copies kept.
- One run, in the log: `[H3MD RH] machine: …` (machine and temp disk) → `removed … leftover cache` (if any) → `template …: … frames (… s), …x…, seamless / segment by segment` → after saving `video saved; segment cache: deleted after save`.

## 5. Contact

YJ (YouTube: AI 纵横 @yj0942). Main repo: <https://github.com/1113800942/comfy-h3md-director>.

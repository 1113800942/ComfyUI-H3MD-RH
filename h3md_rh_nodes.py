"""H3MD RH nodes: the RunningHub build of H3MD Director.

  H3MDDirectorRH      template (MV / motion copy / story / custom) + media from ComfyUI's own Load Image / Load Audio /
                      Load Video nodes (so they show up as inputs of an RH AI app) -> H3MD plan
  H3MDLoopSamplerRH   H3MD Loop Sampler + RefMod, cache always in ComfyUI temp (per-task on RH)
  H3MDSaveVideoRH     H3MD Save (face refine / interpolation / audio), face refine off and cache deleted by default
  H3MDTaraRH          TARA native temporal alignment for motion copy (vendored from ComfyUI-BFSNodes, GPL-3.0,
                      no downloads; see NOTICE.md)

All class names end in RH, so this pack never clashes with ComfyUI-H3MD-Director (any version) installed next to it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import wave

import numpy as np

from .h3md_rh_core import h3md_director as D
from .h3md_rh_core import h3md_director_refmod as R
from .h3md_rh_core import h3md_face_refine_save as FS
from .h3md_rh_core import h3md_window as WIN
from . import h3md_rh_templates as T

LOG = logging.getLogger("H3MD")
_TAG = "[H3MD RH]"
CATEGORY = "MiniMax H3/H3MD RH"


def _temp_dir(sub: str) -> str:
    try:
        import folder_paths
        base = folder_paths.get_temp_directory()
    except Exception:
        base = os.path.join(os.getcwd(), "temp")
    path = os.path.join(base, sub)
    os.makedirs(path, exist_ok=True)
    return path


def _media_dir() -> str:
    return _temp_dir("h3md_rh_media")


def clear_media() -> int:
    """Remove the photo / song / guide-video copies an earlier task left in ComfyUI temp/h3md_rh_media.
    Called by the Director right before it writes this task's copies (a cached Director does not run, so the
    copies its plan points to stay).  Only plain files directly inside that folder are touched."""
    removed = 0
    folder = _media_dir()
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if os.path.isfile(full) and not os.path.islink(full):
            try:
                os.remove(full)
                removed += 1
            except OSError as exc:
                LOG.warning("%s could not remove old media %s: %s", _TAG, name, exc)
    return removed


# ------------------------------------------------------------------------------------------- media -> files
def save_image(image, tag: str) -> str:
    """IMAGE tensor [B,H,W,C] 0..1 -> PNG in temp (first frame); content-addressed, so re-runs reuse it."""
    from PIL import Image
    arr = image[0].detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)[0]
    arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    digest = hashlib.sha1(arr.tobytes() + str(arr.shape).encode()).hexdigest()[:16]
    path = os.path.join(_media_dir(), f"{tag}_{digest}.png")
    if not os.path.isfile(path):
        Image.fromarray(arr).save(path)
    return path


def save_audio(audio, tag: str = "audio") -> str:
    """AUDIO {'waveform': [B,C,N], 'sample_rate'} -> 16-bit WAV in temp (first batch item)."""
    wf = audio["waveform"]
    sr = int(audio["sample_rate"])
    arr = wf[0].detach().cpu().float().numpy() if hasattr(wf, "detach") else np.asarray(wf, dtype=np.float32)[0]
    if arr.ndim == 1:
        arr = arr[None]
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).round().astype("<i2")      # [C, N]
    digest = hashlib.sha1(pcm.tobytes() + f"{sr}/{pcm.shape}".encode()).hexdigest()[:16]
    path = os.path.join(_media_dir(), f"{tag}_{digest}.wav")
    if not os.path.isfile(path):
        with wave.open(path, "wb") as w:
            w.setnchannels(pcm.shape[0])
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.T.tobytes())
    return path


def _file_digest(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def save_video(video, tag: str = "video") -> str:
    """VIDEO (ComfyUI core Load Video) -> file in temp.  A file-backed video is copied as is (no re-encode)."""
    src = None
    try:
        src = video.get_stream_source()
    except Exception:
        src = None
    if isinstance(src, str) and os.path.isfile(src):
        ext = os.path.splitext(src)[1].lower() or ".mp4"
        path = os.path.join(_media_dir(), f"{tag}_{_file_digest(src)}{ext}")
        if not os.path.isfile(path):
            shutil.copyfile(src, path)
        return path
    tmp = os.path.join(_media_dir(), f"{tag}_incoming.mp4")
    video.save_to(tmp)
    path = os.path.join(_media_dir(), f"{tag}_{_file_digest(tmp)}.mp4")
    os.replace(tmp, path)
    return path


def video_size(path: str):
    try:
        import av
        with av.open(path) as c:
            s = c.streams.video[0]
            w, h = int(s.codec_context.width), int(s.codec_context.height)
            rot = 0
            try:
                rot = int(float(s.metadata.get("rotate", 0) or 0)) % 180
            except Exception:
                rot = 0
            return (h, w) if rot == 90 else (w, h)
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------- Director
class H3MDDirectorRH(R.H3MDDirectorRefMod):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": (T.TEMPLATES, {"default": "mv", "tooltip":
                             "mv: photo + song -> music video · motion_copy: copy the motion of a guide video "
                             "(TARA) · story: one shot per line · custom: plan_json as is.\n"
                             "mv：照片+歌=MV · motion_copy：动作复刻 · story：一行一个镜头 · custom：直接用 plan_json"}),
                "who": ("STRING", {"multiline": True, "default": "a young woman with long dark hair, wearing a "
                                   "white summer dress", "tooltip": "Person: look + clothes (English works best). "
                                   "人物：外貌+服装（英文效果最好）"}),
                "scene": ("STRING", {"multiline": True, "default": "a sunny beach with soft waves",
                                     "tooltip": "Where it happens. 场景"}),
                "aspect": (T.ASPECTS, {"default": "16:9", "tooltip": "Picture shape; 'follow video' = the guide "
                                       "video's shape (motion copy). 画面比例"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.3, "max": T.MAX_MEGAPIXELS, "step": 0.05,
                               "tooltip": "Pixels per frame in millions, at most 1.0 (1.0 = about 1344x768). "
                               "每帧像素（百万），最多 1.0"}),
                "music_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1, "tooltip":
                                "mv: the song starts at this second (skip the intro). MV：歌从第几秒开始"}),
                "max_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": T.MAX_SECONDS, "step": 0.5, "tooltip":
                                "Length of the whole video in seconds, at most 30 (0 = 30). mv: shots fill it; "
                                "story: shots are shortened to fit; motion_copy: uses at most this much of the guide. "
                                "整条视频最长几秒，最多 30（0 = 30）"}),
                "shots": ("STRING", {"multiline": True, "default": "", "tooltip":
                          "story: one shot per line (max 8). 故事：一行一个镜头，最多 8 行"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Main photo (face / person). From Load Image. 人物照片"}),
                "image_2": ("IMAGE", {"tooltip": "motion_copy: full-body photo with the outfit (optional). "
                                      "动作复刻：全身照（可选）"}),
                "audio": ("AUDIO", {"tooltip": "mv: the song. From Load Audio. MV：歌"}),
                "video": ("VIDEO", {"tooltip": "motion_copy: the guide video. From Load Video. 动作复刻：原片"}),
                "plan_json": ("STRING", {"multiline": True, "default": "", "tooltip":
                              "custom: an H3MD plan v1 JSON. 自定义计划"}),
                "seamless": ("BOOLEAN", {"default": False, "tooltip":
                             "motion_copy: render the whole shot as one seamless take (CCW context windows, needs the "
                             "core MiniMax H3 context windows; without them it falls back to segments by itself). Off by "
                             "default: with TARA a jump can still show where two windows hand over (11-09 test). "
                             "mv / story are hard cuts and are not affected. "
                             "动作复刻：整段无接缝（接续窗口 CCW）。默认关：和 TARA 一起用时，两个窗口交接处还可能跳一下"
                             "（11-09 实测）。MV、故事不受影响"}),
            },
        }

    RETURN_TYPES = (D.PLAN_TYPE, "INT", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("plan", "segment_count", "total_frames", "summary", "plan_info", "plan_json")
    FUNCTION = "build_rh"
    CATEGORY = CATEGORY
    DESCRIPTION = ("H3MD Director for RunningHub apps: pick a template, connect ComfyUI's Load Image / Load Audio / "
                   "Load Video, fill in who and where. Connect 'plan' to H3MD Loop Sampler RH.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""   # inputs are tensors / plain values: ComfyUI's own input cache is enough

    @staticmethod
    def make_plan(template, who, scene, aspect, megapixels, music_start, max_seconds, shots,
                  image=None, image_2=None, audio=None, video=None, plan_json="", seamless=False):
        raw = H3MDDirectorRH._template_plan(template, who, scene, aspect, megapixels, music_start, max_seconds,
                                            shots, image, image_2, audio, video, plan_json)
        notes = list(raw.pop(T.NOTES_KEY, None) or [])
        if template == "motion_copy" and seamless:
            if core_windows_available():
                raw["windowMode"] = True     # CCW defaults of the engine (chained low-res windows, 243 / 22)
            else:
                notes.append(f"seamless: this ComfyUI ({WIN.comfyui_version()}) has no core MiniMax H3 context windows "
                             f"- rendered segment by segment instead. 无接缝：当前 ComfyUI 没有核心窗口节点，已改用分段式。")
        if notes:
            raw[T.NOTES_KEY] = notes
        return raw

    @staticmethod
    def _template_plan(template, who, scene, aspect, megapixels, music_start, max_seconds, shots,
                       image=None, image_2=None, audio=None, video=None, plan_json=""):
        if template == "custom":
            if not str(plan_json or "").strip():
                raise ValueError(f"{_TAG} template 'custom' needs plan_json. 自定义模板需要填 plan_json。")
            return json.loads(plan_json)
        if image is None:
            raise ValueError(f"{_TAG} connect a photo (Load Image) to 'image'. 请把人物照片接到 image。")
        clear_media()                  # copies from an earlier task; this task's are written next
        img = save_image(image, "img")
        if template == "mv":
            if audio is None:
                raise ValueError(f"{_TAG} mv needs the song: connect Load Audio to 'audio'. MV 需要接歌（audio）。")
            w, h = T.size_for(aspect, megapixels)
            return T.mv(who, scene, img, save_audio(audio, "song"), music_start, w, h, max_seconds)
        if template == "motion_copy":
            if video is None:
                raise ValueError(f"{_TAG} motion_copy needs the guide video: connect Load Video to 'video'. "
                                 f"动作复刻需要接原片（video）。")
            vid = save_video(video, "guide")
            seconds = D.video_stream_seconds(vid)
            if not seconds:
                raise ValueError(f"{_TAG} cannot read the guide video. 原片读不出来。")
            w, h = T.size_for(aspect, megapixels, video_size(vid))
            body = save_image(image_2, "body") if image_2 is not None else None
            return T.motion_copy(who, scene, img, body, vid, seconds, max_seconds, w, h)
        if template == "story":
            w, h = T.size_for(aspect, megapixels)
            return T.story(who, scene, shots, img, w, h, max_seconds)
        raise ValueError(f"{_TAG} unknown template {template!r}")

    def build_rh(self, template, who, scene, aspect, megapixels, music_start, max_seconds, shots,
                 image=None, image_2=None, audio=None, video=None, plan_json="", seamless=False):
        raw = self.make_plan(template, who, scene, aspect, megapixels, music_start, max_seconds, shots,
                             image, image_2, audio, video, plan_json, seamless)
        notes = list(raw.pop(T.NOTES_KEY, None) or [])
        for note in notes:
            LOG.warning("%s %s", _TAG, note)
        text = json.dumps(raw, ensure_ascii=False)
        plan, count, total, summary, info = self.build_refmod(text)
        check_limits(plan, total)     # templates always fit; a custom plan_json is checked the same way
        if notes:
            summary = "\n".join(notes) + "\n" + summary
        LOG.info("%s template %s: %d segment(s), %d frames (%.1f s), %dx%d, %s", _TAG, template, count, total,
                 total / T.FPS, plan.get("width", 0), plan.get("height", 0),
                 "seamless (window mode)" if plan.get("windowMode") else "segment by segment")
        return (plan, count, total, summary, info, text)


MAX_PIXELS = int(T.MAX_MEGAPIXELS * 1_050_000)    # 1 MP + 5 %: sizes snap to multiples of 64 (1344x768 = 1.03 MP)


def check_limits(plan: dict, total: int) -> None:
    """RunningHub limits for every plan (templates and custom plan_json): at most 30 s, at most 1 MP a frame."""
    if total > T.cap_frames(0) + 1:
        raise ValueError(f"{_TAG} the video is {total / T.FPS:.1f} s, over the {T.MAX_SECONDS:g} s limit of this "
                         f"build. 视频 {total / T.FPS:.1f} 秒，超过本版 {T.MAX_SECONDS:g} 秒的上限。")
    w, h = int(plan.get("width") or 0), int(plan.get("height") or 0)
    if w * h > MAX_PIXELS:
        raise ValueError(f"{_TAG} {w}x{h} is over the {T.MAX_MEGAPIXELS:g} MP limit of this build. "
                         f"{w}x{h} 超过本版 {T.MAX_MEGAPIXELS:g} 百万像素的上限。")


def core_windows_available() -> bool:
    """True when this ComfyUI has the core MiniMax H3 context windows that seamless mode (CCW) builds on."""
    try:
        WIN.core_windows_node()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------------------------------- Loop
CACHE_SUB = "h3md_rh_cache"
CACHE_DTYPE = "uint8"          # 0.4.3-rh1: 8-bit frame cache always (half the disk of fp16, no visible difference)
DISK_MARGIN = 1.15             # cache estimate x this + DISK_EXTRA_GB must be free before a run starts
DISK_EXTRA_GB = 0.5            # latents, audio, the saved video


def clear_old_cache(path: str) -> int:
    """Remove what an earlier task left in the RH cache folder (normally the Save node already deleted it).
    Only touches entries inside `path` (ComfyUI temp/h3md_rh_cache).  Returns how many entries were removed."""
    removed = 0
    for name in os.listdir(path):
        full = os.path.join(path, name)
        try:
            if os.path.isdir(full) and not os.path.islink(full):
                shutil.rmtree(full)
            else:
                os.remove(full)
            removed += 1
        except OSError as exc:
            LOG.warning("%s could not remove old cache %s: %s", _TAG, name, exc)
    return removed


def cache_estimate_bytes(plan: dict) -> int:
    """Disk the 8-bit frame cache needs for this plan: every segment's frames at the output size, 3 bytes a pixel."""
    w, h = int(plan.get("width") or 0), int(plan.get("height") or 0)
    frames = sum(int(s.get("length") or 0) for s in plan.get("segments") or [])
    return frames * w * h * 3


def environment() -> dict:
    """ComfyUI version, GPU, VRAM, free RAM - for the log (RH does not show the task machine)."""
    env = {"comfyui": WIN.comfyui_version(), "gpu": "none", "vram_gb": 0.0, "ram_free_gb": None}
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env["gpu"], env["vram_gb"] = p.name, round(p.total_memory / 2 ** 30, 1)
    except Exception:
        pass
    try:
        import psutil
        env["ram_free_gb"] = round(psutil.virtual_memory().available / 2 ** 30, 1)
    except Exception:
        pass
    return env


def preflight(plan: dict, cache_dir: str) -> dict:
    """Log the machine and check the temp disk before anything is encoded; raise a clear error when the cache will
    not fit (instead of failing half way)."""
    env = environment()
    need = cache_estimate_bytes(plan) * DISK_MARGIN + DISK_EXTRA_GB * 2 ** 30
    free = shutil.disk_usage(cache_dir).free
    env.update(disk_free_gb=round(free / 2 ** 30, 1), disk_need_gb=round(need / 2 ** 30, 1))
    LOG.info("%s machine: ComfyUI %s, GPU %s (%.1f GB), free RAM %s GB, temp disk free %.1f GB; this run needs about "
             "%.1f GB of temp disk in %s (%d frames at %dx%d, 8-bit cache, %s)", _TAG, env["comfyui"], env["gpu"],
             env["vram_gb"], env["ram_free_gb"], env["disk_free_gb"], env["disk_need_gb"], cache_dir,
             sum(int(s.get("length") or 0) for s in plan.get("segments") or []), int(plan.get("width") or 0),
             int(plan.get("height") or 0), "seamless" if plan.get("windowMode") else "segment by segment")
    if free < need:
        raise RuntimeError(
            f"{_TAG} not enough temp disk for this video: needs about {env['disk_need_gb']} GB, "
            f"{env['disk_free_gb']} GB free. Make the video shorter (max_seconds) or smaller (megapixels). "
            f"临时盘空间不够：这条视频大约要 {env['disk_need_gb']} GB，只剩 {env['disk_free_gb']} GB。"
            f"请缩短时长（max_seconds）或降低分辨率（megapixels）。")
    return env


class H3MDLoopSamplerRH(R.H3MDLoopSamplerRefMod):
    CATEGORY = CATEGORY
    DESCRIPTION = ("H3MD Loop Sampler for RunningHub: same as H3MD Loop Sampler + RefMod. The segment cache always "
                   "lives in ComfyUI temp/h3md_rh_cache, is 8-bit, and what an earlier task left there is removed "
                   "first; the temp disk is checked before the run starts. Use stop_after to run only the first "
                   "segments.")

    def run(self, options=None, **kwargs):
        opts = dict(options) if isinstance(options, dict) else {}
        if not str(opts.get("cache_dir") or "").strip():
            opts["cache_dir"] = _temp_dir(CACHE_SUB)
        if opts.get("cache_dtype", CACHE_DTYPE) != CACHE_DTYPE:
            LOG.info("%s cache_dtype %s -> %s (the RH build always caches 8-bit)", _TAG, opts["cache_dtype"],
                     CACHE_DTYPE)
        opts["cache_dtype"] = CACHE_DTYPE
        resume = bool(kwargs.get("resume_from_cache", False))      # main-node input (not in Loop Options)
        if not resume and os.path.realpath(opts["cache_dir"]) == os.path.realpath(_temp_dir(CACHE_SUB)):
            n = clear_old_cache(opts["cache_dir"])
            if n:
                LOG.info("%s removed %d leftover cache folder(s) from an earlier task", _TAG, n)
        plan = kwargs.get("plan")
        if isinstance(plan, dict):
            preflight(plan, opts["cache_dir"])
        return super().run(options=opts, **kwargs)


# --------------------------------------------------------------------------------------------------- Save
class H3MDSaveVideoRH(FS.H3MDSaveVideoFaceRefineInterp):
    CATEGORY = CATEGORY
    DESCRIPTION = ("H3MD Save for RunningHub: joins the segments, optional face refine (off by default) and RIFE "
                   "interpolation, deletes the segment cache after saving. 'video' output = standard VIDEO.")

    @classmethod
    def INPUT_TYPES(cls):
        spec = copy.deepcopy(FS.H3MDSaveVideoFaceRefineInterp.INPUT_TYPES())
        for sect in ("required", "optional"):
            for name, default in (("face_refine", False), ("delete_cache_after_save", True),
                                  ("filename_prefix", "H3MD_RH")):
                if name in spec.get(sect, {}):
                    kind, opts = spec[sect][name][0], dict(spec[sect][name][1] if len(spec[sect][name]) > 1 else {})
                    opts["default"] = default
                    spec[sect][name] = (kind, opts)
        return spec

    def save(self, *args, **kwargs):
        out = super().save(*args, **kwargs)
        ui = out.get("ui", {}) if isinstance(out, dict) else {}
        state = (ui.get("h3md_cache_state") or ["?"])[0]
        LOG.info("%s video saved; segment cache: %s (%s)", _TAG, state, (ui.get("h3md_cache_dir") or ["?"])[0])
        return out


# --------------------------------------------------------------------------------------------------- TARA
def _tara_class():
    from .h3md_rh_tara import minimax_h3_native_temporal as NT

    AUTO = "auto (flash / kitchen / flex)"

    class H3MDTaraRH(NT.BFSMiniMaxH3NativeTemporal):
        CATEGORY = CATEGORY
        DESCRIPTION = ("TARA native temporal alignment for motion copy (H3MD RH build of BFSMiniMaxH3NativeTemporal "
                       "by Alisson Anjos, ComfyUI-BFSNodes, GPL-3.0). Put it on the LOW-RES model line, after the "
                       "TARA motion-transfer LoRA, with no Sage attention on that line. Never downloads anything.")

        @classmethod
        def INPUT_TYPES(cls):
            spec = copy.deepcopy(NT.BFSMiniMaxH3NativeTemporal.INPUT_TYPES())
            spec["optional"]["attention_backend"] = ([AUTO] + list(NT.BACKENDS), {"default": AUTO, "tooltip":
                "auto: flash partitions when this PyTorch build has flash attention (Linux / RunningHub); else comfy "
                "kitchen frame-wise when comfy-kitchen is installed (Windows); else flex (needs torch.compile). "
                "自动：flash → kitchen → flex"})
            return spec

        def apply(self, model, *args, attention_backend=AUTO, **kwargs):
            if attention_backend == AUTO:
                attention_backend = auto_backend(NT)
                LOG.info("%s TARA backend: %s", _TAG, attention_backend)
            return super().apply(model, *args, attention_backend=attention_backend, **kwargs)

    return H3MDTaraRH, NT


_FLASH = None


def auto_backend(NT) -> str:
    if flash_ok():
        return NT.BACKENDS[0]
    try:
        from .h3md_rh_tara import minimax_h3_temporal_kitchen as K
        K.kitchen_function()
        return NT.BACKENDS[2]
    except Exception:
        return NT.BACKENDS[1]


def flash_ok() -> bool:
    """True when this PyTorch build runs the flash SDPA kernel (Windows wheels usually do not)."""
    global _FLASH
    if _FLASH is None:
        try:
            import torch
            if not torch.cuda.is_available():
                _FLASH = False
            else:
                q = torch.zeros(1, 1, 8, 64, device="cuda", dtype=torch.bfloat16)
                torch.ops.aten._scaled_dot_product_flash_attention(q, q, q)
                _FLASH = True
        except Exception:
            _FLASH = False
    return _FLASH


NODE_CLASS_MAPPINGS = {
    "H3MDDirectorRH": H3MDDirectorRH,
    "H3MDLoopSamplerRH": H3MDLoopSamplerRH,
    "H3MDSaveVideoRH": H3MDSaveVideoRH,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MDDirectorRH": "H3MD Director RH",
    "H3MDLoopSamplerRH": "H3MD Loop Sampler RH",
    "H3MDSaveVideoRH": "H3MD Save Video RH",
}
try:
    _Tara, _NT = _tara_class()
    NODE_CLASS_MAPPINGS["H3MDTaraRH"] = _Tara
    NODE_DISPLAY_NAME_MAPPINGS["H3MDTaraRH"] = "H3MD TARA RH (motion copy, from BFSNodes)"
except Exception as _exc:   # never blocks the other nodes
    LOG.warning("%s TARA node not loaded: %s", _TAG, _exc)

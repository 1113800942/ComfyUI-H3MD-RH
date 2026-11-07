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
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 4.0, "step": 0.05, "tooltip":
                               "Pixels per frame in millions (1.0 = about 1344x768). Higher costs more on RH. "
                               "每帧像素（百万），越高越贵"}),
                "music_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1, "tooltip":
                                "mv: the song starts at this second (skip the intro). MV：歌从第几秒开始"}),
                "max_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.5, "tooltip":
                                "motion_copy: use at most this many seconds of the guide (0 = all). "
                                "动作复刻：最多用原片几秒（0=全部）"}),
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
                  image=None, image_2=None, audio=None, video=None, plan_json=""):
        if template == "custom":
            if not str(plan_json or "").strip():
                raise ValueError(f"{_TAG} template 'custom' needs plan_json. 自定义模板需要填 plan_json。")
            return json.loads(plan_json)
        if image is None:
            raise ValueError(f"{_TAG} connect a photo (Load Image) to 'image'. 请把人物照片接到 image。")
        img = save_image(image, "img")
        if template == "mv":
            if audio is None:
                raise ValueError(f"{_TAG} mv needs the song: connect Load Audio to 'audio'. MV 需要接歌（audio）。")
            w, h = T.size_for(aspect, megapixels)
            return T.mv(who, scene, img, save_audio(audio, "song"), music_start, w, h)
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
            return T.story(who, scene, shots, img, w, h)
        raise ValueError(f"{_TAG} unknown template {template!r}")

    def build_rh(self, template, who, scene, aspect, megapixels, music_start, max_seconds, shots,
                 image=None, image_2=None, audio=None, video=None, plan_json=""):
        raw = self.make_plan(template, who, scene, aspect, megapixels, music_start, max_seconds, shots,
                             image, image_2, audio, video, plan_json)
        text = json.dumps(raw, ensure_ascii=False)
        plan, count, total, summary, info = self.build_refmod(text)
        LOG.info("%s template %s: %d segment(s), %d frames, %dx%d", _TAG, template, count, total,
                 plan.get("width", 0), plan.get("height", 0))
        return (plan, count, total, summary, info, text)


# --------------------------------------------------------------------------------------------------- Loop
class H3MDLoopSamplerRH(R.H3MDLoopSamplerRefMod):
    CATEGORY = CATEGORY
    DESCRIPTION = ("H3MD Loop Sampler for RunningHub: same as H3MD Loop Sampler + RefMod, the segment cache always "
                   "lives in ComfyUI temp (per task on RH). Use stop_after to run only the first segments.")

    def run(self, options=None, **kwargs):
        opts = dict(options) if isinstance(options, dict) else {}
        if not str(opts.get("cache_dir") or "").strip():
            opts["cache_dir"] = _temp_dir("h3md_rh_cache")
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

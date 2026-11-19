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



def guide_camera_moves(path: str, seconds: float = None, fps: float = 4.0, lag_s: float = 2.0):
    """How much the guide's camera moves: the median change of the frame border (outer sixth on the left / right,
    top sixth) between frames 2 s apart, 64x36 grey, median over time.  The person in the middle, or an arm crossing
    the border, moves only part of it; a moving camera changes all of it.  Measured 11-09: tracking / moving cameras
    16-19, fixed-camera dances 2-6 (threshold T.CAMERA_MOVES = 10).  PyAV only (no external program).  None when the
    clip cannot be read or is shorter than 2 s."""
    try:
        import av
        frames, step, nxt = [], 1.0 / fps, 0.0
        with av.open(path) as c:
            st = c.streams.video[0]
            st.thread_type = "AUTO"
            for fr in c.decode(st):
                t = float(fr.time or 0.0)
                if seconds and t > float(seconds):
                    break
                if t + 1e-6 < nxt:
                    continue
                nxt = t + step
                frames.append(fr.to_ndarray(format="gray", width=64, height=36).astype(np.float32))
        lag = int(round(lag_s * fps))
        if len(frames) <= lag:
            return None
        a = np.stack(frames)
        m = np.zeros(a.shape[1:], bool)
        m[:, :64 // 6] = m[:, -(64 // 6):] = m[:36 // 6, :] = True
        d = np.abs(a[lag:] - a[:-lag])[:, m]
        return float(np.median(np.median(d, axis=1)))
    except Exception as exc:
        LOG.info("%s guide camera check skipped: %s", _TAG, exc)
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
                "scene": ("STRING", {"multiline": True, "default": "", "tooltip":
                          "Where it happens (empty: mv uses a sunny beach). motion_copy: leave EMPTY to keep the guide "
                          "video's own surroundings (steadiest, any guide); a scene replaces them - with a guide whose "
                          "camera moves this can jump where two windows hand over (the summary says so).\n"
                          "场景（留空：MV 用晴天海滩）。动作复刻：留空 = 用原片的景物（最稳，什么原片都行）；填了 = 换成新场景，"
                          "原片镜头在动时两个窗口交接处可能跳（摘要里会提醒）"}),
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
                "video": ("VIDEO", {"tooltip": "motion_copy: the guide video, ONE continuous shot (no cuts inside); the "
                                    "camera may move. From Load Video. 动作复刻：原片，要是一个连续镜头（中间不要有剪辑），"
                                    "镜头可以动"}),
                "plan_json": ("STRING", {"multiline": True, "default": "", "tooltip":
                              "custom: an H3MD plan v1 JSON. 自定义计划"}),
                "seamless": ("BOOLEAN", {"default": True, "tooltip":
                             "motion_copy: render the whole shot as one seamless take (CCW context windows, needs the "
                             "core MiniMax H3 context windows; without them it falls back to segments by itself). "
                             "Tested 11-09 with a fixed-camera dance and a tracking run. mv / story are hard cuts and are "
                             "not affected. 动作复刻：整段无接缝（接续窗口 CCW），固定机位的舞和跟拍的跑步都测过"
                             "（11-09）。MV、故事不受影响"}),
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
                  image=None, image_2=None, audio=None, video=None, plan_json="", seamless=True):
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
            moves = guide_camera_moves(vid, min(float(seconds), T.cap_seconds(max_seconds)))
            plan = T.motion_copy(who, scene, img, body, vid, seconds, max_seconds, w, h,
                                 camera_moves=None if moves is None else moves >= T.CAMERA_MOVES)
            LOG.info("%s motion copy: guide camera score %s (moves at %.0f or more: tracking shot), %s", _TAG,
                     "n/a" if moves is None else f"{moves:.1f}", T.CAMERA_MOVES,
                     "new scene" if T.clean(scene) else "the guide's own surroundings")
            note = T.camera_note(moves, scene)
            if note:
                plan[T.NOTES_KEY] = list(plan.get(T.NOTES_KEY) or []) + [note]
            return plan
        if template == "story":
            w, h = T.size_for(aspect, megapixels)
            return T.story(who, scene, shots, img, w, h, max_seconds)
        raise ValueError(f"{_TAG} unknown template {template!r}")

    def build_rh(self, template, who, scene, aspect, megapixels, music_start, max_seconds, shots,
                 image=None, image_2=None, audio=None, video=None, plan_json="", seamless=True):
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


# 0.5.5-rh1: Remake (multi-person swap) only runs at these sizes; 1664x960 / 960x1664 are allowed on RH even though
# they are over 1 MP (2560x1472 is not: too much VRAM / temp disk for an RH task).
SWAP_SIZES = ((1664, 960), (960, 1664))


def is_swap(plan: dict) -> bool:
    return str(plan.get("swapEngine") or "").strip().lower() == "remake"


def check_limits(plan: dict, total: int) -> None:
    """RunningHub limits for every plan (templates, custom plan_json, imported plans): at most 30 s; at most 1 MP a
    frame, except a multi-person swap plan at 1664x960 / 960x1664 (the Remake sizes)."""
    if total > T.cap_frames(0) + 1:
        raise ValueError(f"{_TAG} the video is {total / T.FPS:.1f} s, over the {T.MAX_SECONDS:g} s limit of this "
                         f"build. 视频 {total / T.FPS:.1f} 秒，超过本版 {T.MAX_SECONDS:g} 秒的上限。")
    w, h = int(plan.get("width") or 0), int(plan.get("height") or 0)
    if is_swap(plan) and (w, h) not in SWAP_SIZES:
        raise ValueError(f"{_TAG} a multi-person swap plan must be 1664x960 or 960x1664 on RunningHub (got {w}x{h}). "
                         f"多人换人计划在 RH 上只能是 1664x960 或 960x1664（现在是 {w}x{h}）。")
    if w * h > MAX_PIXELS and not (is_swap(plan) and (w, h) in SWAP_SIZES):
        raise ValueError(f"{_TAG} {w}x{h} is over the {T.MAX_MEGAPIXELS:g} MP limit of this build. "
                         f"{w}x{h} 超过本版 {T.MAX_MEGAPIXELS:g} 百万像素的上限。")


def core_windows_available() -> bool:
    """True when this ComfyUI has the core MiniMax H3 context windows that seamless mode (CCW) builds on."""
    try:
        WIN.core_windows_node()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------------------- Plan Import
# 0.5.5-rh1.  On RunningHub a plan zip is uploaded with RH's own "Load Images From Zip V2" (RHUploadZip_V2): RH stores
# the zip and unpacks it to input/zip_extracts/<hash>/ on the machine that runs the task (tested 2026-11-18), and an
# AI app can expose its upload box.  This node reads that folder - no web route, no front-end script.  (The main
# pack's "Import plan..." button does not work on RH: it unpacks on the web machine, which the task machine never
# sees - the pictures came up blank.)
PLAN_NAME = "plan.json"
SOURCE_TOL_S = 0.1                 # an uploaded source must match the plan's source length within this
MAX_PREVIEW = 16


def find_plan_file(folder: str) -> str:
    """plan.json at the top of the unpacked zip, or the only one below it (a zip of a folder)."""
    top = os.path.join(folder, PLAN_NAME)
    if os.path.isfile(top):
        return top
    found = []
    for root, _dirs, files in os.walk(folder):
        if PLAN_NAME in files:
            found.append(os.path.join(root, PLAN_NAME))
    if len(found) == 1:
        return found[0]
    if not found:
        raise ValueError(f"{_TAG} no plan.json in the uploaded zip - export the plan from H3MD Director "
                         f"('Export plan + media') and upload that zip. 上传的 zip 里没有 plan.json，请用导演台的"
                         f"「Export plan + media」导出计划包再上传。")
    raise ValueError(f"{_TAG} the zip holds {len(found)} plan.json files - upload one plan at a time. "
                     f"zip 里有 {len(found)} 个 plan.json，一次只传一个计划包。")


def _index_files(folder: str) -> dict:
    """lower-case base name -> path, for every media file in the unpacked zip (first one wins)."""
    from .h3md_rh_core import h3md_plan_io as PIO
    out = {}
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(PIO.MEDIA_EXTS):
                out.setdefault(name.lower(), os.path.join(root, name))
    return out


def relink_media(plan: dict, folder: str) -> list:
    """Point every media file of the plan at its copy in the unpacked zip (matched by file name, like the main
    pack's import).  Returns the files that are not in the zip."""
    have = _index_files(folder)
    missing = []
    for kind in ("images", "videos", "audios"):
        for it in (plan.get("media") or {}).get(kind) or []:
            if not isinstance(it, dict) or not it.get("file"):
                continue
            base = os.path.basename(str(it["file"]).replace("\\", "/")).lower()
            if base in have:
                it["file"] = have[base]
            else:
                missing.append((kind, str(it.get("id") or ""), str(it["file"])))
    return missing


def source_entry(plan: dict):
    """The plan's source video (the one an uploaded video replaces): the only video, else id 'src', else the video
    the locked soundtrack comes from.  None when the plan has no video."""
    vids = [v for v in (plan.get("media") or {}).get("videos") or [] if isinstance(v, dict)]
    if not vids:
        return None
    if len(vids) == 1:
        return vids[0]
    by_id = {str(v.get("id")): v for v in vids}
    if "src" in by_id:
        return by_id["src"]
    for s in plan.get("segments") or []:
        src = str(((s or {}).get("locked") or {}).get("source") or "")
        if src.startswith("video:") and src[6:] in by_id:
            return by_id[src[6:]]
    raise ValueError(f"{_TAG} the plan has {len(vids)} videos and none is marked as the source (id 'src') - "
                     f"leave 'video' unconnected so the zip's own videos are used. 计划里有 {len(vids)} 个视频，"
                     f"分不出哪个是原片；请不要接 video，直接用计划包里的视频。")


def plan_source_seconds(plan: dict, entry: dict):
    if entry.get("duration"):
        return float(entry["duration"])
    vid = str(entry.get("id"))
    uses = [float(v.get("useDuration")) for s in plan.get("segments") or [] for v in (s or {}).get("videos") or []
            if isinstance(v, dict) and str(v.get("id")) == vid and v.get("useDuration")]
    return max(uses) if uses else None


def resolve_trim(plan: dict, start_s: float, end_s: float, uploaded_s, wanted_s):
    """(start, end) to cut from the uploaded video, or None to use it as it is.
    - start_s / end_s on the node win when either is set;
    - else an uploaded clip that already has the plan's length is used as it is (the user uploaded the cut part);
    - else the plan's sourceTrim (written by H3MD Keyframe 1: Prepare from main 0.5.6);
    - else as it is (the plan used the whole source)."""
    start_s, end_s = max(0.0, float(start_s or 0.0)), max(0.0, float(end_s or 0.0))
    if start_s or end_s:
        return start_s, end_s
    if wanted_s and uploaded_s and abs(uploaded_s - wanted_s) <= SOURCE_TOL_S:
        return None
    st = plan.get("sourceTrim")
    if isinstance(st, dict) and (st.get("start") or st.get("end")):
        return max(0.0, float(st.get("start") or 0.0)), max(0.0, float(st.get("end") or 0.0))
    return None


def cut_video(video, start_s: float, end_s: float, tag: str = "source") -> str:
    """Core VIDEO -> the part start_s .. end_s (0 = to the end) as a file in temp/h3md_rh_media."""
    if end_s and end_s <= start_s:
        raise ValueError(f"{_TAG} end_s ({end_s:g}) must be after start_s ({start_s:g}), or 0 for the end. "
                         f"end_s 要大于 start_s，或填 0 表示到结尾。")
    if not hasattr(video, "as_trimmed"):
        raise ValueError(f"{_TAG} cutting a part of the video needs a newer ComfyUI (Trim Video). Upload the cut part "
                         f"instead. 截取需要较新的 ComfyUI；请直接上传截好的那一段。")
    part = video.as_trimmed(start_s, (end_s - start_s) if end_s else 0.0)
    if part is None:
        raise ValueError(f"{_TAG} could not cut {start_s:g}-{end_s or 'end'} s from the video. "
                         f"原片截不出 {start_s:g}-{end_s or '结尾'} 秒这一段。")
    # not save_video(): a trimmed VideoFromFile still reports the whole original as its stream source, so a plain
    # file copy would keep the whole clip; save_to() writes only the trimmed part
    tmp = os.path.join(_media_dir(), f"{tag}_cut_incoming.mp4")
    part.save_to(tmp)
    path = os.path.join(_media_dir(), f"{tag}_{start_s:g}-{end_s:g}s_{_file_digest(tmp)}.mp4")
    os.replace(tmp, path)
    return path


def load_preview(paths: list):
    """Pictures -> IMAGE batch [N,H,W,3] at the first picture's size (for a Preview Image node)."""
    import torch
    from PIL import Image
    frames, size = [], None
    for p in paths[:MAX_PREVIEW]:
        try:
            im = Image.open(p).convert("RGB")
        except Exception as exc:
            LOG.warning("%s preview skipped %s: %s", _TAG, os.path.basename(p), exc)
            continue
        size = size or im.size
        if im.size != size:
            im = im.resize(size, Image.LANCZOS)
        frames.append(np.asarray(im, dtype=np.float32) / 255.0)
    if not frames:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    return torch.from_numpy(np.stack(frames))


class H3MDPlanImportRH(R.H3MDDirectorRefMod):
    """Uploaded H3MD plan zip (+ optional source video) -> H3MD plan, checked against the RH limits."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"forceInput": True, "tooltip":
                                "Connect 'folder_path' of RunningHub's Load Images From Zip V2 (RHUploadZip_V2), "
                                "where the plan zip is uploaded. 接 RH「Load Images From Zip V2」的 folder_path（在那里上传计划包 zip）"}),
            },
            "optional": {
                "video": ("VIDEO", {"tooltip":
                          "The source video (Load Video). Replaces the plan's source; leave unconnected to use the "
                          "video inside the zip. The whole original is fine: the part the keyframes were made from is "
                          "cut by itself (the plan's sourceTrim, or start_s / end_s). 原片（加载视频）。会替换计划里的原片；"
                          "不接就用 zip 里的。传整条原片也行，会自动截出做关键帧的那一段"}),
                "start_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1, "tooltip":
                            "Cut the uploaded video from this second (0 and end_s 0 = automatic: the plan's sourceTrim, "
                            "or the video as it is). 原片从第几秒截（和 end_s 都是 0 = 自动）"}),
                "end_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1, "tooltip":
                          "... to this second (0 = to the end). 截到第几秒（0 = 到结尾）"}),
                "global_prompt": ("STRING", {"multiline": True, "default": "", "tooltip":
                                  "Optional: trigger words / style for every shot. Empty = keep the plan's own. "
                                  "可选：每个镜头都加的触发词/风格；留空 = 用计划自己的"}),
            },
        }

    RETURN_TYPES = (D.PLAN_TYPE, "INT", "INT", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("plan", "segment_count", "total_frames", "summary", "plan_info", "keyframes")
    FUNCTION = "import_rh"
    CATEGORY = CATEGORY
    DESCRIPTION = ("H3MD Plan Import for RunningHub: upload a plan zip made on your own PC (keyframes, prompts, "
                   "settings) with RH's Load Images From Zip V2, connect its folder_path here, optionally the source "
                   "video from Load Video. Connect 'plan' to H3MD Loop Sampler RH, 'keyframes' to Preview Image.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ""

    def import_rh(self, folder_path, video=None, start_s=0.0, end_s=0.0, global_prompt=""):
        from .h3md_rh_core import h3md_paths as PATHS
        from .h3md_rh_core import h3md_plan_io as PIO
        folder = str(folder_path or "").strip().strip('"')
        if not folder or not os.path.isdir(folder):
            raise ValueError(f"{_TAG} the plan zip is not there ({folder or 'empty'}): upload it in Load Images From "
                             f"Zip V2 first. 找不到计划包（{folder or '空'}）：请先在 Load Images From Zip V2 上传。")
        PATHS.check_read(folder)
        with open(PATHS.check_read(find_plan_file(folder)), encoding="utf-8-sig") as f:
            plan = PIO._load_plan(f.read())
        notes = []
        missing = relink_media(plan, folder)
        src = source_entry(plan)
        if video is not None:
            if src is None:
                raise ValueError(f"{_TAG} this plan has no source video, so the uploaded video has nowhere to go - "
                                 f"leave 'video' unconnected. 这个计划没有原片，请不要接 video。")
            clear_media()
            wanted = plan_source_seconds(plan, src)
            path = save_video(video, "source")
            got = D.video_stream_seconds(path)
            trim = resolve_trim(plan, start_s, end_s, got, wanted)
            if trim:
                path = cut_video(video, trim[0], trim[1])
                got = D.video_stream_seconds(path)
                notes.append(f"source video: cut {trim[0]:g}-{trim[1] or 'end'} s of the uploaded video")
            if wanted and got and abs(got - wanted) > SOURCE_TOL_S:
                raise ValueError(
                    f"{_TAG} the source video does not match the plan: it is {got:.2f} s, the keyframes were made on "
                    f"{wanted:.2f} s. Upload the same part, or set start_s / end_s to the part you cut on your PC. "
                    f"原片对不上：现在是 {got:.2f} 秒，关键帧是在 {wanted:.2f} 秒的片段上做的。请上传同一段，"
                    f"或把 start_s / end_s 填成你在本地截的那一段。")
            src["file"] = path
            missing = [m for m in missing if not (m[0] == "videos" and m[1] == str(src.get("id")))]
        if missing:
            names = ", ".join(f"{os.path.basename(m[2])} ({m[0][:-1]} {m[1]})" for m in missing)
            raise ValueError(f"{_TAG} not in the zip: {names}. Export with 'Export plan + media'"
                             f"{' or connect the source video' if any(m[0] == 'videos' for m in missing) else ''}. "
                             f"zip 里缺这些文件：{names}。请用「Export plan + media」导出"
                             f"{'，或接上原片' if any(m[0] == 'videos' for m in missing) else ''}。")
        if str(global_prompt or "").strip():
            plan["globalPrompt"] = str(global_prompt).strip()
        if is_swap(plan):
            w, h = int(plan.get("width") or 0), int(plan.get("height") or 0)
            if (w, h) not in SWAP_SIZES:
                nw, nh = (960, 1664) if h > w else (1664, 960)
                notes.append(f"multi-person swap: {w}x{h} -> {nw}x{nh} (the Remake size allowed on RunningHub)")
                plan["width"], plan["height"] = nw, nh
        problems = [p for p in PIO.check_plan(plan) if not p.startswith("missing file")]
        if problems:
            notes += [f"check: {p}" for p in problems]
        for n in notes:
            LOG.warning("%s %s", _TAG, n)
        out, count, total, summary, info = self.build_refmod(json.dumps(plan, ensure_ascii=False))
        check_limits(out, total)
        if notes:
            summary = "\n".join(notes) + "\n" + summary
        pics = [str(it["file"]) for it in (plan.get("media") or {}).get("images") or []
                if isinstance(it, dict) and it.get("file")]
        LOG.info("%s plan imported from %s: %d segment(s), %d frames (%.1f s), %dx%d, %d picture(s)%s", _TAG,
                 os.path.basename(folder), count, total, total / T.FPS, out.get("width", 0), out.get("height", 0),
                 len(pics), ", multi-person swap (Remake)" if is_swap(plan) else "")
        return (out, count, total, summary, info, load_preview(pics))


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
    "H3MDPlanImportRH": H3MDPlanImportRH,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MDDirectorRH": "H3MD Director RH",
    "H3MDLoopSamplerRH": "H3MD Loop Sampler RH",
    "H3MDSaveVideoRH": "H3MD Save Video RH",
    "H3MDPlanImportRH": "H3MD Plan Import RH 计划包导入",
}
try:
    _Tara, _NT = _tara_class()
    NODE_CLASS_MAPPINGS["H3MDTaraRH"] = _Tara
    NODE_DISPLAY_NAME_MAPPINGS["H3MDTaraRH"] = "H3MD TARA RH (motion copy, from BFSNodes)"
except Exception as _exc:   # never blocks the other nodes
    LOG.warning("%s TARA node not loaded: %s", _TAG, _exc)

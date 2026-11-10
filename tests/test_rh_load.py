"""Load ComfyUI-H3MD-Director (main) and ComfyUI-H3MD-RH side by side; build plans for every template."""
import importlib, importlib.util, json, math, os, sys, tempfile, wave
ROOT = os.environ.get("COMFYUI_ROOT", "/home/claude/ComfyUI")
sys.path.insert(0, ROOT); sys.dont_write_bytecode = True
import comfy.cli_args; comfy.cli_args.args.cpu = True
import folder_paths, torch, numpy as np
folder_paths.set_temp_directory(tempfile.mkdtemp()) if hasattr(folder_paths, "set_temp_directory") else None

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m

main = load("main_h3md", "/home/claude/comfy-h3md-director")
rh = load("h3md_rh", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
clash = set(main.NODE_CLASS_MAPPINGS) & set(rh.NODE_CLASS_MAPPINGS)
print("main nodes", len(main.NODE_CLASS_MAPPINGS), "rh nodes", sorted(rh.NODE_CLASS_MAPPINGS))
assert not clash, clash
assert set(rh.NODE_CLASS_MAPPINGS) == {"H3MDDirectorRH", "H3MDLoopSamplerRH", "H3MDSaveVideoRH", "H3MDTaraRH"}
for k, c in rh.NODE_CLASS_MAPPINGS.items():
    c.INPUT_TYPES()

N = sys.modules["h3md_rh.h3md_rh_nodes"]; T = sys.modules["h3md_rh.h3md_rh_templates"]
img = torch.rand(1, 512, 384, 3)
aud = {"waveform": torch.sin(torch.linspace(0, 4000, 48000 * 90)).reshape(1, 1, -1) * 0.3, "sample_rate": 48000}
d = N.H3MDDirectorRH()
base = dict(who='a woman in a "red" dress\nwith {braces}', scene="a beach", aspect="16:9", megapixels=1.0,
            music_start=16.0, max_seconds=0.0, shots="")
plan, n, total, summ, info, text = d.build_rh("mv", image=img, audio=aud, **base)
raw = json.loads(text)
print("mv", n, total, plan["width"], plan["height"], [s["audios"][0]["offset"] for s in raw["segments"]][:3])
# 0.4.3-rh1: 30 s cap -> 3 shots of 226 frames (9.42 s), song offsets follow the shot length
assert n == 3 and [s["length"] for s in raw["segments"]] == [226] * 3 and total <= 720, (n, total)
assert raw["segments"][1]["audios"][0]["offset"] == round(16.0 + 226 / 24, 3)
assert "windowMode" not in raw, "mv is hard cuts only: no window mode"
assert '"red"' not in raw["segments"][0]["prompt"] and "{braces}" not in raw["segments"][0]["prompt"]

# guide video: 7.5 s 480x720 with audio
import av
vp = os.path.join(tempfile.mkdtemp(), "guide.mp4")
with av.open(vp, "w") as c:
    s = c.add_stream("libx264", rate=30); s.width, s.height, s.pix_fmt = 480, 720, "yuv420p"
    a = c.add_stream("aac", rate=48000)
    for i in range(225):
        fr = av.VideoFrame.from_ndarray(np.full((720, 480, 3), i % 255, np.uint8), format="rgb24")
        for p in s.encode(fr): c.mux(p)
    for p in s.encode(): c.mux(p)
    for i in range(int(7.5 * 48000 / 1024)):
        af = av.AudioFrame.from_ndarray(np.zeros((1, 1024), np.float32), format="fltp", layout="mono"); af.sample_rate = 48000
        for p in a.encode(af): c.mux(p)
    for p in a.encode(): c.mux(p)
class V:
    def get_stream_source(self): return vp
plan, n, total, summ, info, text = d.build_rh("motion_copy", image=img, image_2=img, video=V(),
                                              **dict(base, aspect="follow video"))
assert plan.get("windowMode"), "rh2: seamless is on by default (11-09 T1c fixed camera + T1d tracking run joined cleanly)"
raw = json.loads(text)
print("motion_copy", n, total, plan["width"], plan["height"], raw["segments"][0]["length"], summ.splitlines()[:3])
assert raw["segments"][0]["length"] == 175 and (total - 5) % 17 == 0 and total <= 180
assert plan["width"] < plan["height"]

plan, n, total, summ, info, text = d.build_rh("story", image=img, **dict(base, shots="she walks in\n\nshe sits down\nshe smiles"))
print("story", n, total); assert n == 3
try:
    d.build_rh("mv", image=img, **base); raise SystemExit("mv without audio should fail")
except ValueError as e:
    print("ok error:", str(e)[:60])

# ---------------------------------------------------------------- 0.4.3-rh1 additions
import shutil as _sh
# caps: max_seconds above 30 -> 30; megapixels above 1 -> 1 MP
plan, n, total, *_ = d.build_rh("mv", image=img, audio=aud, **dict(base, max_seconds=45.0, megapixels=4.0))
assert total <= 720 and plan["width"] * plan["height"] <= 1_050_000, (total, plan["width"], plan["height"])
plan, n, total, *_ = d.build_rh("mv", image=img, audio=aud, **dict(base, max_seconds=20.0))
assert n == 2 and total <= 480, (n, total)
# story: 7 lines do not fit in 30 s at >= 4.5 s each -> first 6 kept, summary says so
plan, n, total, summ, info, text = d.build_rh("story", image=img, **dict(base, shots="\n".join(f"shot {i}" for i in range(7))))
assert n == 6 and total <= 720 and "dropped" in summ, (n, total, summ[:200])
print("caps ok: story 7 lines ->", n, "shots,", total, "frames")

# motion copy: seamless on -> window mode in the plan; off -> segments
plan, n, total, summ, info, text = d.build_rh("motion_copy", image=img, image_2=img, video=V(), seamless=True,
                                              **dict(base, aspect="follow video"))
assert json.loads(text).get("windowMode") is True and plan.get("windowMode"), "seamless must write windowMode"
assert json.loads(text)["refVideoHeight"] == "480" and plan["refVideoHeight"] == "480", "guide video at short edge 480"
assert (plan["windowMode"] or {}).get("chain") is True
plan_off, *_ = d.build_rh("motion_copy", image=img, image_2=img, video=V(), seamless=False,
                          **dict(base, aspect="follow video"))
assert not plan_off.get("windowMode")
# no core context windows -> falls back to segments, says so, no error
_real = N.core_windows_available
N.core_windows_available = lambda: False
try:
    plan_fb, n_fb, total_fb, summ_fb, *_ = d.build_rh("motion_copy", image=img, image_2=img, video=V(), seamless=True,
                                                      **dict(base, aspect="follow video"))
finally:
    N.core_windows_available = _real
assert not plan_fb.get("windowMode") and "segment by segment" in summ_fb, summ_fb[:200]
print("seamless ok: on -> window mode, off / no core node -> segments")

# rh2 (11-09): motion copy follows the guide's camera; empty scene = the guide's own surroundings; a new scene with a
# guide whose camera moves gets a note (YJ 16:07: keep the scene, say so)
assert N.H3MDDirectorRH.INPUT_TYPES()["required"]["scene"][1]["default"] == ""
assert N.H3MDDirectorRH.INPUT_TYPES()["optional"]["seamless"][1]["default"] is True
def mc_prompt(r): return r["segments"][0]["prompt"]
plan, n, total, summ, info, text = d.build_rh("motion_copy", image=img, image_2=img, video=V(),
                                              **dict(base, aspect="follow video", scene=""))
pr = mc_prompt(json.loads(text))
assert "locked-off" not in pr and "dances in" not in pr and "stay the same from start to end" not in pr, pr
assert "surroundings are the ones in <Video 1>" in pr and "tracking shot that moves exactly like the camera of <Video 1>" in pr
# moving guide + empty scene = word for word the prompt that joined cleanly in T1d (11-09)
_t1d = T.motion_copy("a young woman, wearing the outfit shown in the photo", "", "f", "b", "v", 12.25, 0.0, 1344, 768,
                     camera_moves=True)["segments"][0]["prompt"]
assert _t1d == open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "t1d_prompt.txt"),
                    encoding="utf-8").read(), "moving guide + empty scene must stay the T1d prompt"
_still = T.motion_copy("w", "", "f", None, "v", 12.25, camera_moves=False)["segments"][0]["prompt"]
assert "tracking" not in _still and "when the guide's camera stands still, it stands still" in _still
assert "camera moves" not in summ.split("H3MD plan")[0], summ[:300]          # empty scene: no note
plan, n, total, summ, info, text = d.build_rh("motion_copy", image=img, image_2=img, video=V(),
                                              **dict(base, aspect="follow video", scene="a rooftop at sunset"))
pr = mc_prompt(json.loads(text))
assert "The setting is a rooftop at sunset" in pr and "its setting is replaced by a rooftop at sunset" in pr
assert "surroundings are the ones in <Video 1>" not in pr
score = N.guide_camera_moves(vp)
assert score is not None and score > T.CAMERA_MOVES, score     # the test guide changes every pixel every frame
assert "the guide's camera moves" in summ and "原片镜头在动" in summ, summ[:400]
# a fixed camera: still border, something moving in the middle -> low score, no note even with a scene
vp2 = os.path.join(tempfile.mkdtemp(), "still.mp4")
with av.open(vp2, "w") as c:
    s2 = c.add_stream("libx264", rate=24); s2.width, s2.height, s2.pix_fmt = 384, 216, "yuv420p"
    bg = np.tile(np.linspace(40, 200, 384, dtype=np.uint8)[None, :, None], (216, 1, 3))
    for i in range(24 * 6):
        f = bg.copy(); x = 120 + int(60 * math.sin(i / 5)); f[60:160, x:x + 60] = 255 - (i * 7) % 255
        for p in s2.encode(av.VideoFrame.from_ndarray(f, format="rgb24")): c.mux(p)
    for p in s2.encode(): c.mux(p)
score2 = N.guide_camera_moves(vp2)
assert score2 is not None and score2 < T.CAMERA_MOVES, score2
assert T.camera_note(score2, "a rooftop") is None and T.camera_note(score, "") is None and T.camera_note(None, "x") is None
print(f"rh2 motion copy ok: follow / new scene prompts, camera score moving {score:.0f} still {score2:.0f}, note only when moving + scene")

# Loop RH: 8-bit cache forced, leftover cache removed, temp disk checked before the run
LS = sys.modules["h3md_rh.h3md_rh_core.h3md_loop_split"]
seen = {}
_orig_run = LS.H3MDLoopSampler2.run
LS.H3MDLoopSampler2.run = lambda self, options=None, **kw: seen.update(options=options, kw=kw) or ("ran",)
cache_root = N._temp_dir(N.CACHE_SUB)
os.makedirs(os.path.join(cache_root, "old_task_seg"), exist_ok=True)
open(os.path.join(cache_root, "old_task_seg", "f.npy"), "wb").write(b"x" * 10)
try:
    out = N.H3MDLoopSamplerRH().run(options={"cache_dtype": "fp16"}, plan=plan, resume_from_cache=False)
    assert out == ("ran",) and seen["options"]["cache_dtype"] == "uint8", seen["options"]
    assert seen["options"]["cache_dir"] == cache_root and not os.listdir(cache_root), os.listdir(cache_root)
    # resume on: leftovers kept
    os.makedirs(os.path.join(cache_root, "keep_me"), exist_ok=True)
    N.H3MDLoopSamplerRH().run(options=None, plan=plan, resume_from_cache=True)
    assert os.listdir(cache_root) == ["keep_me"]
    # not enough disk -> clear error before anything runs
    _du = N.shutil.disk_usage
    N.shutil.disk_usage = lambda p: _sh._ntuple_diskusage(100, 99, 1)
    try:
        N.H3MDLoopSamplerRH().run(options=None, plan=plan, resume_from_cache=True)
        raise SystemExit("preflight should fail on a full disk")
    except RuntimeError as e:
        assert "temp disk" in str(e) and "临时盘" in str(e)
        print("ok preflight error:", str(e)[:90])
    finally:
        N.shutil.disk_usage = _du
finally:
    LS.H3MDLoopSampler2.run = _orig_run
    _sh.rmtree(cache_root, ignore_errors=True)
# custom plan_json gets the same limits: 40 s or 1920x1088 -> clear error
st_text = d.build_rh("story", image=img, **dict(base, shots="one line"))[5]
big = json.loads(st_text); big["segments"][0]["length"] = 960
for bad, why in ((dict(big), "s limit"), (dict(json.loads(st_text), width=1920, height=1088), "MP limit")):
    try:
        d.build_rh("custom", **dict(base, plan_json=json.dumps(bad))); raise SystemExit("custom over limit must fail")
    except ValueError as e:
        assert why in str(e), str(e)
print("custom plan limits ok")
# Director: media copies of an earlier task are replaced, this task's copies exist
md = N._media_dir(); open(os.path.join(md, "img_oldtask.png"), "wb").write(b"x")
plan, n, total, summ, info, text = d.build_rh("story", image=img, **dict(base, shots="one line"))
left = os.listdir(md)
assert "img_oldtask.png" not in left and any(f.startswith("img_") for f in left), left
assert all(os.path.isfile(m["file"]) for m in json.loads(text)["media"]["images"])
est = N.cache_estimate_bytes({"width": 1920, "height": 1088, "segments": [{"length": 1}]})
assert est == 1920 * 1088 * 3     # 6.3 MB per 1920x1088 frame, as measured on 11-08
print("loop RH ok: uint8 forced, leftovers cleared, disk checked")
# Save RH: a run folder holding run.json (the run's base seed) is deleted after saving (11-09 T1b: it was kept)
SS = sys.modules["h3md_rh.h3md_rh_core.h3md_stream_save"]
rd = os.path.join(tempfile.mkdtemp(), "run_20261109_145623_10012_83503")
os.makedirs(os.path.join(rd, "seg_000_progress"))
for f in ("run.json", "seg_000.json", "seg_000_frames.npy"):
    open(os.path.join(rd, f), "wb").write(b"x")
assert SS.delete_cache_dir(rd) and not os.path.exists(rd)
print("save RH ok: run cache with run.json deleted")
print("ALL OK")

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
assert n == 8 and raw["segments"][1]["audios"][0]["offset"] == 26.125
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
print("ALL OK")

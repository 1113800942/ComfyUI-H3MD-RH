"""H3MD Plan Import RH (0.5.5-rh1): a plan zip unpacked the way RunningHub does it (input/zip_extracts/<hash>/),
optional source video from Load Video.  Needs COMFYUI_ROOT, the main repo at /home/claude/comfy-h3md-director and
the demo swap plan zip in H3MD_DEMO_ZIP (example_plans/H3MD_demo_multiswap_remake.zip of the main repo)."""
import importlib.util, json, os, shutil, sys, tempfile, zipfile
ROOT = os.environ.get("COMFYUI_ROOT", "/home/claude/ComfyUI")
DEMO = os.environ.get("H3MD_DEMO_ZIP", "/tmp/demo_swap.zip")
sys.path.insert(0, ROOT); sys.dont_write_bytecode = True
import comfy.cli_args; comfy.cli_args.args.cpu = True
import folder_paths, numpy as np, av
tmp_root = tempfile.mkdtemp()
folder_paths.set_temp_directory(os.path.join(tmp_root, "temp"))
folder_paths.set_input_directory(os.path.join(tmp_root, "input"))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m

rh = load("h3md_rh", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N = sys.modules["h3md_rh.h3md_rh_nodes"]
from comfy_api.latest._input_impl.video_types import VideoFromFile
node = N.H3MDPlanImportRH()

def unpack(zpath, tag, edit=None, drop=(), prefix=""):
    """RH style: input/zip_extracts/<hash>/ (optionally edit plan.json / drop members / nest in a folder)."""
    dest = os.path.join(folder_paths.get_input_directory(), "zip_extracts", tag)
    shutil.rmtree(dest, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if n in drop:
                continue
            data = z.read(n)
            if n == "plan.json" and edit:
                p = json.loads(data); edit(p); data = json.dumps(p).encode()
            out = os.path.join(dest, prefix, n); os.makedirs(os.path.dirname(out), exist_ok=True)
            open(out, "wb").write(data)
    return dest

def err(fn):
    try:
        fn()
    except Exception as e:
        return str(e)
    raise AssertionError("no error")

# 1. demo swap plan, no video connected: the zip's own source and keyframes
f1 = unpack(DEMO, "h1")
plan, n, total, summ, info, kf = node.import_rh(f1)
print("1", n, total, plan["width"], plan["height"], tuple(kf.shape), summ.splitlines()[0])
assert (plan["width"], plan["height"]) == (1664, 960) and total == 243 and kf.shape[0] == 3
assert plan.get("swapEngine") == "remake"
files = json.dumps(plan.get("media", {}), default=str)
assert "zip_extracts" in files, "media must point into the unpacked zip"

# source clip of the demo (10.125 s) and a longer 'original': 4 s of grey + the clip + 3 s of grey
src_clip = os.path.join(f1, "media", "swap5_src.mp4")
def make_long(path):
    with av.open(src_clip) as ci:
        vs = ci.streams.video[0]; w, h = vs.codec_context.width, vs.codec_context.height
        frames = [fr.to_ndarray(format="rgb24") for fr in ci.decode(vs)]
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=24); s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        au = c.add_stream("aac", rate=48000)
        grey = np.full((h, w, 3), 128, np.uint8)
        for a in [grey] * 96 + frames + [grey] * 72:
            for p in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")): c.mux(p)
        for p in s.encode(): c.mux(p)
        for i in range(int((96 + len(frames) + 72) / 24 * 48000 / 1024)):
            af = av.AudioFrame.from_ndarray((np.sin(np.arange(1024, dtype=np.float32) / 9) * 0.2)[None], format="fltp", layout="mono")
            af.sample_rate = 48000
            for p in au.encode(af): c.mux(p)
        for p in au.encode(): c.mux(p)
    return len(frames)
long_mp4 = os.path.join(tmp_root, "original.mp4"); nfr = make_long(long_mp4)
print("clip frames", nfr)

# 2. the cut clip itself uploaded: used as it is
plan2, *_ = node.import_rh(f1, video=VideoFromFile(src_clip))
v2 = [v["file"] for v in json.loads(json.dumps(plan2["media"]["videos"]))] if isinstance(plan2["media"]["videos"], list) else None
print("2 ok", str(plan2["media"]["videos"])[:160])

# 3. whole original + plan sourceTrim (main 0.5.6 writes it): cut automatically
f3 = unpack(DEMO, "h3", edit=lambda p: p.update(sourceTrim={"start": 4.0, "end": round(4.0 + nfr / 24, 4)}))
plan3, n3, t3, summ3, *_ = node.import_rh(f3, video=VideoFromFile(long_mp4))
print("3", summ3.splitlines()[0]); assert "cut 4-" in summ3
cut = [v for v in plan3["media"]["videos"].values()][0]["file"] if isinstance(plan3["media"]["videos"], dict) else None
with av.open(cut) as c:
    kinds = [s.type for s in c.streams]
print("3 cut file", os.path.basename(cut), kinds); assert "audio" in kinds, "the cut source must keep its sound (locked audio)"

# 4. whole original, old plan without sourceTrim, start_s / end_s typed on the node
plan4, *_ = node.import_rh(f1, video=VideoFromFile(long_mp4), start_s=4.0, end_s=round(4.0 + nfr / 24, 4))
# 5. whole original, nothing to say which part: clear error
e5 = err(lambda: node.import_rh(f1, video=VideoFromFile(long_mp4))); print("5", e5[:120]); assert "does not match" in e5

# 6. no plan.json / zip of a folder / missing keyframe / 2560x1472 swap -> 1664x960 / non-swap over 1 MP
e6 = err(lambda: node.import_rh(unpack(DEMO, "h6", drop=("plan.json",)))); assert "no plan.json" in e6
plan7, *_ = node.import_rh(unpack(DEMO, "h7", prefix="my plan")); assert plan7["width"] == 1664
e8 = err(lambda: node.import_rh(unpack(DEMO, "h8", drop=("media/kf_0096.png",)))); print("8", e8[:100]); assert "kf_0096.png" in e8
plan9, _, _, summ9, *_ = node.import_rh(unpack(DEMO, "h9", edit=lambda p: p.update(width=2560, height=1472)))
assert (plan9["width"], plan9["height"]) == (1664, 960) and "2560x1472 -> 1664x960" in summ9
def noswap(p): p.pop("swapEngine"); p.update(width=1664, height=960)
e10 = err(lambda: node.import_rh(unpack(DEMO, "h10", edit=noswap))); print("10", e10[:90]); assert "MP limit" in e10
# 11. global prompt override; empty keeps the plan's
plan11, *_ = node.import_rh(f1, global_prompt="cinematic film look")
assert "cinematic film look" in json.dumps(plan11, default=str)
e12 = err(lambda: node.import_rh(os.path.join(folder_paths.get_input_directory(), "zip_extracts", "nope"))); assert "not there" in e12
print("PLAN IMPORT ALL OK")

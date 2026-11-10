"""Build the three RunningHub demo workflows (UI format) for H3MD RH.

Usage: python tools/make_rh_workflows.py <object_info.json> <out dir>
  object_info.json: GET /object_info of a ComfyUI that has this pack, KJNodes and YCNodes-MiniMax-H3 installed.
  Every template runs the whole video (stop_after 0): the Director caps it at 30 s and 1 MP.

Flat graphs (no subgraphs), LoRAs on core LoraLoaderModelOnly nodes so they can be re-picked on RH (file names
differ there).  Group 1 holds everything a RH AI app should expose.
"""
import json
import os
import sys

BASE = {   # tested settings (drum TARA full run, 2026-11-05 / RH tests 2026-11-06)
    "unet": "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors",
    "vae_audio": "minimax_h3_audio_vae_fp32.safetensors",
    "vae_video": "minimax_h3_video_vae_fp16.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "turbo": "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors",
    "tara": "minimax_h3_motion_transfer_tara_v1.0_ref2va.safetensors",
    "combat": "H3_Combat_V2.safetensors",
    "enhancer": "MiniMaxH3_Enhancer_0.5-1.0_3432130_epoch_10.safetensors",
    "realism": "h3-realism-people-r34l1sm-1.0 strong 0.6-0.8 light.safetensors",
    "lms": "minimax_h3_lms_v1.0_r64.safetensors",
    "upscaler": "h3_upscaler_sharpness_2000steps_v0.1_fp32.safetensors",
}


VERSION = "0.4.3-rh2"


class Graph:
    def __init__(self, info):
        self.info, self.nodes, self.links, self.groups = info, [], [], []

    def _widgets(self, ntype):
        spec = self.info[ntype]
        order = spec.get("input_order") or {s: list(spec["input"].get(s, {})) for s in ("required", "optional")}
        out = []
        for sect in ("required", "optional"):
            for name in order.get(sect, []):
                v = spec["input"][sect][name]
                t, opt = v[0], (v[1] if len(v) > 1 else {})
                if isinstance(t, list) or t in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"):
                    if opt.get("forceInput"):
                        continue
                    out.append((name, t, opt))
        return out

    def add(self, ntype, pos, values=None, title=None, size=(340, 120)):
        values = values or {}
        spec = self.info[ntype]
        nid = len(self.nodes) + 1
        wv = []
        for name, t, opt in self._widgets(ntype):
            if name in values:
                val = values[name]
            elif "default" in opt:
                val = opt["default"]
            elif isinstance(t, list) and t:
                val = t[0]
            elif t == "COMBO" and opt.get("options"):
                val = opt["options"][0]
            else:
                val = None
            wv.append(val)
            if opt.get("control_after_generate"):
                wv.append(values.get(name + "__control", "fixed"))
            if opt.get("image_upload") or opt.get("video_upload"):
                wv.append("image")
            if opt.get("audio_upload"):
                wv += [None, None]
        node = {"id": nid, "type": ntype, "pos": list(pos), "size": list(size), "flags": {}, "order": nid - 1,
                "mode": 0, "inputs": [], "outputs": [
                    {"name": n, "type": t, "links": [], "slot_index": i}
                    for i, (t, n) in enumerate(zip(spec["output"], spec.get("output_name", spec["output"])))],
                "properties": {"Node name for S&R": ntype}, "widgets_values": wv}
        if title:
            node["title"] = title
        self.nodes.append(node)
        return nid

    def link(self, src, slot, dst, name):
        s, d = self.nodes[src - 1], self.nodes[dst - 1]
        typ = s["outputs"][slot]["type"]
        lid = len(self.links) + 1
        s["outputs"][slot]["links"].append(lid)
        d["inputs"].append({"name": name, "type": typ, "link": lid})
        self.links.append([lid, src, slot, dst, len(d["inputs"]) - 1, typ])

    def group(self, title, box, color):
        self.groups.append({"title": title, "bounding": list(box), "color": color, "font_size": 28})

    def dump(self, path, note):
        wf = {"last_node_id": len(self.nodes), "last_link_id": len(self.links), "nodes": self.nodes,
              "links": self.links, "groups": self.groups, "config": {},
              "extra": {"h3md_rh": note}, "version": 0.4}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wf, f, ensure_ascii=False, indent=1)


def build(info, kind, inputs):
    g = Graph(info)
    X0, Y0 = 0, 0
    # ---------------- group 1: user inputs (expose these in the RH AI app)
    img = g.add("LoadImage", (X0, Y0), {"image": inputs["image"]}, "人物照片 Photo", (320, 360))
    img2 = aud = vid = None
    if kind == "motion_copy":
        img2 = g.add("LoadImage", (X0, Y0 + 400), {"image": inputs["image_2"]}, "全身照 Full body (可选)", (320, 360))
        vid = g.add("LoadVideo", (X0, Y0 + 800), {"file": inputs["video"]}, "参考动作视频 Guide video", (320, 360))
    if kind == "mv":
        aud = g.add("LoadAudio", (X0, Y0 + 400), {"audio": inputs["audio"]}, "歌曲 Song", (320, 140))
    director = g.add("H3MDDirectorRH", (X0 + 360, Y0), {
        "template": kind, "who": inputs["who"], "scene": inputs["scene"], "aspect": inputs["aspect"],
        "megapixels": 1.0, "music_start": inputs.get("music_start", 0.0), "max_seconds": inputs.get("max_seconds", 0.0),
        "shots": inputs.get("shots", ""), "seamless": True},
        "H3MD Director RH（填人物 / 场景；最长 30 秒、1 MP）", (420, 660))
    g.link(img, 0, director, "image")
    if img2:
        g.link(img2, 0, director, "image_2")
    if vid:
        g.link(vid, 0, director, "video")
    if aud:
        g.link(aud, 0, director, "audio")
    g.group("① 用户输入 User inputs（AI 应用开放这里）", (X0 - 20, Y0 - 60, 840, 1260), "#3f789e")

    # ---------------- group 2: models + LoRAs (re-pick LoRA names on RH)
    MX, MY = X0 + 900, Y0
    unet = g.add("UNETLoader", (MX, MY), {"unet_name": BASE["unet"], "weight_dtype": "default"}, None, (360, 90))
    clip = g.add("CLIPLoader", (MX, MY + 130), {"clip_name": BASE["clip"], "type": "minimax"}, None, (360, 110))
    vae = g.add("VAELoader", (MX, MY + 270), {"vae_name": BASE["vae_video"]}, "视频 VAE", (360, 60))
    avae = g.add("VAELoader", (MX, MY + 360), {"vae_name": BASE["vae_audio"]}, "音频 VAE", (360, 60))
    lines = {}
    for li, (which, chunks) in enumerate((("low", 2), ("high", 3))):
        x = MX + 420 + li * 420
        prev = g.add("MiniMaxChunkFeedForward", (x, MY), {"chunks": chunks, "seq_threshold": 4096},
                     f"{'低清' if which == 'low' else '高清'} chunk FF", (360, 90))
        g.link(unet, 0, prev, "model")
        chain = []
        if kind == "motion_copy":
            chain.append(("tara", 1.0, "TARA 动作迁移 LoRA"))
            chain.append(("combat", 0.7 if which == "low" else 0.49, "Combat V2"))
        if which == "low":
            chain += [("enhancer", 0.5, "Enhancer"), ("turbo", 1.0, "加速 turbo 768 1.2")]
        else:
            chain += [("realism", 0.5, "Realism"), ("lms", 0.2, "LMS"), ("turbo", 0.5, "加速 turbo 768 1.2")]
        for k, (key, st, label) in enumerate(chain):
            n = g.add("LoraLoaderModelOnly", (x, MY + 130 + k * 120), {"lora_name": BASE[key], "strength_model": st},
                      f"{'低清' if which == 'low' else '高清'} {label}", (360, 90))
            g.link(prev, 0, n, "model")
            prev = n
        if which == "low" and kind == "motion_copy":
            n = g.add("H3MDTaraRH", (x, MY + 130 + len(chain) * 120),
                      {"attention_backend": "auto (flash / kitchen / flex)"}, "H3MD TARA RH（只接低清，低清不开 Sage）",
                      (360, 250))
            g.link(prev, 0, n, "model")
            prev = n
        if which == "high":
            n = g.add("MiniMaxH3MemoryEfficientSageAttentionPatch", (x, MY + 130 + len(chain) * 120), {},
                      "高清 Sage", (360, 50))
            g.link(prev, 0, n, "model")
            prev = n
        lines[which] = prev
    g.group("② 模型和 LoRA（RH 上按 RH 的文件名重新选）", (MX - 20, MY - 60, 1300, 1000), "#8a8a3f")

    # ---------------- group 3: generation (do not change)
    GX, GY = X0, Y0 + 1300
    samp = g.add("KSamplerSelect", (GX, GY), {"sampler_name": "euler"}, None, (300, 60))
    sched = g.add("BasicScheduler", (GX, GY + 100), {"scheduler": "simple", "steps": 8, "denoise": 1.0}, None,
                  (300, 110))
    g.link(lines["low"], 0, sched, "model")
    refine = g.add("H3SigmaRefiner", (GX, GY + 250), {"extra_steps": 1, "start_at_sigma": 0.7, "end_at_sigma": 0.0,
                                                     "spacing": "cosine"}, None, (300, 130))
    g.link(sched, 0, refine, "sigmas")
    loop = g.add("H3MDLoopSamplerRH", (GX + 360, GY), {
        "stop_after": inputs["stop_after"], "seed": inputs.get("seed", 23793874848978), "seed__control": "randomize",
        "upscaler_model": BASE["upscaler"], "seed_mode": "increment"}, "H3MD Loop Sampler RH（8 位缓存在临时目录）",
        (420, 620))
    for src, slot, name in ((director, 0, "plan"), (clip, 0, "clip"), (vae, 0, "vae"), (avae, 0, "audio_vae"),
                            (lines["low"], 0, "low_res_model"), (samp, 0, "sampler"), (refine, 0, "sigmas"),
                            (lines["high"], 0, "high_res_model")):
        g.link(src, slot, loop, name)
    save = g.add("H3MDSaveVideoRH", (GX + 820, GY), {"filename_prefix": f"H3MD_RH_{kind}",
                                                    "detector": "face_yolov8n_v2.pt"}, "H3MD Save Video RH", (420, 700))
    g.link(loop, 5, save, "segments")
    g.group("③ 生成（勿改）", (GX - 20, GY - 60, 1300, 800), "#444")
    return g


DEMOS = {
    "mv": dict(image="eve_meadow_full.png", audio="rhtest_water.mp3", aspect="16:9", music_start=16.0,
               who="a young East Asian woman with long dark hair, wearing the same outfit as in the photo",
               scene="a sunny tropical beach with turquoise water and soft waves", stop_after=0),
    "motion_copy": dict(image="face.png", image_2="full.png", video="guide.mp4", aspect="follow video",
                        max_seconds=0.0, who="a young woman, wearing the outfit shown in the photo",
                        scene="", stop_after=0),   # empty = the guide's own surroundings (rh2)
    "story": dict(image="eve_meadow_full.png", aspect="16:9",
                  who="a young East Asian woman with long dark hair, wearing the same outfit as in the photo",
                  scene="a quiet seaside town in the morning",
                  shots="She walks slowly along the harbour wall, looking out at the boats.\n"
                        "She stops at a small cafe and smiles at the camera.\n"
                        "She sits at a table by the window with a cup of coffee, the sea behind her.",
                  stop_after=0),
}

if __name__ == "__main__":
    info = json.load(open(sys.argv[1], encoding="utf-8"))
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    names = {"mv": "H3MD RH MV 照片加歌", "motion_copy": "H3MD RH 动作复刻", "story": "H3MD RH 多镜头故事"}
    for kind, inputs in DEMOS.items():
        g = build(info, kind, inputs)
        path = os.path.join(out, names[kind] + ".json")
        g.dump(path, {"kind": kind, "version": VERSION})
        print(path, len(g.nodes), "nodes", len(g.links), "links")

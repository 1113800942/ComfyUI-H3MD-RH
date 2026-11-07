"""Native Kitchen extension lookup - H3MD RH build.

MODIFIED from ComfyUI-BFSNodes minimax_h3_temporal_kernel.py (GPL-3.0, Alisson Anjos), 2026-11-06, by YJ:
the first-use download of prebuilt binaries (minimax_h3_temporal_bootstrap) is removed, so this build never
downloads code.  The Kitchen one-pass kernel is used only when comfy_kitchen.temporal_attention is already
installed; otherwise api_for() returns None and the callers fall back to their PyTorch paths.
"""
import importlib


def api_for(q):
    if q.device.type != 'cuda':
        return None
    try:
        api = importlib.import_module('comfy_kitchen.temporal_attention')
    except ImportError:
        return None
    if getattr(api, 'VERSION', None) != 'h3_native_temporal_kernel_v1':
        return None
    return api if api.is_available(q.device) else None


def policy_for(api, plan, ctx):
    key=(api.VERSION,ctx.strength,ctx.window,ctx.falloff,ctx.cap)
    if getattr(plan,'_native_temporal_policy_key',None)!=key:
        times = getattr(plan, 'frame_times', ())
        if not times:
            times = tuple(ctx.times[0, plan.target_start:plan.target_stop:plan.target_frame_rows].detach().cpu().tolist())
        plan._native_temporal_policy = api.TemporalPolicy(
            plan.source_start, plan.source_stop, plan.target_start, plan.target_stop,
            plan.source_frame_rows, plan.target_frame_rows, tuple(times),
            ctx.strength, ctx.window, ctx.falloff, ctx.cap)
        plan._native_temporal_policy_key=key
    return plan._native_temporal_policy


def run(kind, q, k, v, plan, ctx, **kwargs):
    api = api_for(q)
    if api is None:
        return None
    policy = policy_for(api, plan, ctx)
    if not getattr(plan, '_native_'+kind+'_logged', False):
        print(f'[H3MD TARA RH] Kitchen ONE-PASS {kind} kernel ACTIVE (installed comfy_kitchen extension).', flush=True)
        setattr(plan, '_native_'+kind+'_logged', True)
    dtype=q.dtype
    if str(dtype)=='torch.float32':
        import torch
        q,k,v=[t.to(torch.bfloat16) for t in (q,k,v)]
    fn = api.int8_attention if kind == 'dense' else api.sol_attn
    return fn(q,k,v,policy,**kwargs).to(dtype)


def prepare_model(model):
    """Prepare a selected native backend during node execution, before sampling."""
    device=getattr(model,'load_device',None)
    if getattr(device,'type',None)=='cuda':
        from types import SimpleNamespace
        return api_for(SimpleNamespace(device=device))
    return None

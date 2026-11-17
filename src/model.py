# ============================================================
# Model side: text loaders (6 domains), memory-light forward
# hook attention extraction O(T^2), and real pruned-cache decoding.
# ============================================================
import glob
import os
import sysconfig
import numpy as np
from src.core import FullCache, HeavyHitter


def _hf_text(name, config, split, fn, need, cap, sep="\n\n"):
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split, streaming=True)
    parts, n = [], 0
    for ex in ds:
        s = fn(ex)
        if not s:
            continue
        s = s[:cap]
        parts.append(s)
        n += len(s)
        if n >= need:
            break
    return sep.join(parts)


def _stdlib_code(need, cap):
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "*.py"))) + sorted(glob.glob(os.path.join(root, "*", "*.py")))
    files = [f for f in files if "test" not in f.lower() and "site-packages" not in f and "idlelib" not in f]
    parts, n = [], 0
    for f in files:
        try:
            s = open(f, encoding="utf-8").read()[:cap]
        except Exception:
            continue
        if len(s) < 2000:
            continue
        parts.append(s)
        n += len(s)
        if n >= need:
            break
    return "\n\n".join(parts)


TEXT_LOADERS = {
    "wikipedia": lambda need, cap: _hf_text(
        "Salesforce/wikitext", "wikitext-103-raw-v1", "test", lambda e: e["text"], need, cap, sep=""
    ),
    "code": _stdlib_code,
    "dialogue": lambda need, cap: _hf_text(
        "HuggingFaceH4/ultrachat_200k", None, "test_sft",
        lambda e: "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in e["messages"]),
        need, cap,
    ),
    "books": lambda need, cap: _hf_text("emozilla/pg19-test", None, "test", lambda e: e["text"], need, cap),
    "science": lambda need, cap: _hf_text("ccdv/arxiv-summarization", "document", "test", lambda e: e["article"], need, cap),
    "math": lambda need, cap: _hf_text(
        "openai/gsm8k", "main", "test", lambda e: f"Question: {e['question']}\nAnswer: {e['answer']}", need, cap
    ),
}


def load_model(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=getattr(torch, cfg["dtype"]), attn_implementation="eager"
    ).to(dev).eval()
    return tok, model


def attention_matrix(model, ids):
    """One forward pass with the full cache. Returns A (T x T, mean over layers and heads, rows sum to 1)
    and nll[t] (loss predicting ids[t+1]). Per-layer attention is summed on the fly by hooks and then
    discarded, so memory is O(T^2) instead of O(layers * heads * T^2)."""
    import torch
    import torch.nn.functional as F
    dev = model.device
    T = len(ids)
    acc = torch.zeros(T, T, device=dev, dtype=torch.float32)
    n = [0]
    mods = [m for nm, m in model.named_modules() if nm.endswith("self_attn")]

    def hook(mod, args, out):
        if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
            acc.add_(out[1][0].float().mean(0))
            n[0] += 1
            return (out[0], None) + tuple(out[2:])

    hs = [m.register_forward_hook(hook) for m in mods]
    try:
        with torch.no_grad():
            x = torch.as_tensor(ids, dtype=torch.long, device=dev)[None]
            out = model(x, output_attentions=True, use_cache=False)
            nll = F.cross_entropy(out.logits[0, :-1].float(), x[0, 1:], reduction="none")
    finally:
        for h in hs:
            h.remove()
    if n[0] == 0:
        raise RuntimeError("no attention weights captured: load the model with attn_implementation='eager'")
    A = (acc / n[0]).cpu().numpy().astype(np.float64)
    A = np.tril(A)
    A /= A.sum(1, keepdims=True)
    return A, nll.cpu().numpy()


def _prune(past, keep_idx):
    if isinstance(past, tuple):
        return tuple((k[:, :, keep_idx], v[:, :, keep_idx]) for k, v in past)
    if hasattr(past, "key_cache"):
        for i in range(len(past.key_cache)):
            past.key_cache[i] = past.key_cache[i][:, :, keep_idx]
            past.value_cache[i] = past.value_cache[i][:, :, keep_idx]
        if hasattr(past, "_seen_tokens"):
            past._seen_tokens = past.key_cache[0].shape[2]
        return past
    for layer in past.layers:
        layer.keys = layer.keys[:, :, keep_idx]
        layer.values = layer.values[:, :, keep_idx]
    return past


def run_with_eviction(model, ids, k, policy, gen_tokens=0):
    """Decode token by token, physically deleting evicted KV entries (kept tokens keep their original
    RoPE positions). Returns nll[t] (predicting ids[t+1]) and an optional greedy continuation."""
    import torch
    import torch.nn.functional as F
    from transformers import DynamicCache
    dev = model.device
    ids = torch.as_tensor(ids, dtype=torch.long)
    T = len(ids)
    policy.reset(T, k)
    need = policy.uses_attention
    cache, past = [], DynamicCache()
    nll = np.full(T - 1, np.nan)
    logits = None
    with torch.no_grad():
        for t in range(T):
            cache.append(t)
            ev = policy.select_evictions(t, cache)
            if ev:
                evs = set(ev)
                keep = [i for i, j in enumerate(cache[:-1]) if j not in evs]
                past = _prune(past, torch.tensor(keep, device=dev))
                cache = [j for j in cache if j not in evs]
            assert len(cache) <= k
            out = model(
                ids[t:t + 1][None].to(dev),
                past_key_values=past,
                position_ids=torch.tensor([[t]], device=dev),
                attention_mask=torch.ones(1, len(cache), device=dev, dtype=torch.long),
                use_cache=True,
                output_attentions=need,
            )
            past = out.past_key_values
            logits = out.logits[0, -1].float()
            if t < T - 1:
                nll[t] = -F.log_softmax(logits, -1)[ids[t + 1]].item()
            if need:
                a = torch.stack([x[0, :, -1].float().mean(0) for x in out.attentions]).mean(0).cpu().numpy()
                policy.observe(t, cache, a / a.sum())
        gen, pos = [], T
        for _ in range(gen_tokens):
            nxt = int(logits.argmax())
            gen.append(nxt)
            out = model(
                torch.tensor([[nxt]], device=dev),
                past_key_values=past,
                position_ids=torch.tensor([[pos]], device=dev),
                attention_mask=torch.ones(1, len(cache) + len(gen), device=dev, dtype=torch.long),
                use_cache=True,
            )
            past, logits = out.past_key_values, out.logits[0, -1].float()
            pos += 1
    return nll, gen


def model_selftest(tok, model, cfg):
    """Hook-extracted attention is causal & normalised; decoding with a full cache reproduces the
    full forward pass; decoding with eviction respects capacity."""
    ids = tok("The quick brown fox jumps over the lazy dog. " * 8, add_special_tokens=False)["input_ids"][:64]
    A, nll = attention_matrix(model, ids)
    assert np.allclose(A.sum(1), 1) and np.allclose(np.triu(A, 1), 0), "attention matrix malformed"
    nll2, _ = run_with_eviction(model, ids, len(ids), FullCache())
    tol = 1e-2 if cfg["dtype"] == "float32" else 0.25
    diff = float(np.max(np.abs(nll - nll2)))
    assert diff < tol, f"pruned-cache decoding does not match the full forward pass (max diff {diff:.4f})"
    run_with_eviction(model, ids, 16, HeavyHitter(0.5, "sum", "H2O"))
    return diff

# ============================================================
# Model side: (1) extract the full-cache attention matrix A,
# (2) run the model token-by-token with a REAL pruned KV cache
#     controlled by a policy, measuring next-token loss.
# Written for transformers==4.46.x (pinned in the notebook).
# ============================================================
import numpy as np
import torch
import torch.nn.functional as F
try:
    from transformers import DynamicCache
except ImportError:
    DynamicCache = None


@torch.no_grad()
def attention_matrix(model, ids):
    """ids: 1-D LongTensor of length T. Returns A (T x T numpy) = attention
    averaged over all layers and heads, plus per-token NLL with full cache."""
    out = model(ids[None].to(model.device), output_attentions=True)
    att = torch.stack([a[0].float().mean(0) for a in out.attentions]).mean(0)   # T x T
    logp = F.log_softmax(out.logits[0].float(), -1)
    nll = -logp[:-1].gather(1, ids[1:, None].to(model.device))[:, 0]          # nll[t] predicts t+1
    A = att.cpu().numpy().astype(np.float64)
    A /= A.sum(1, keepdims=True)
    return A, nll.cpu().numpy()


def _prune(past, keep_idx):
    """Keep cache positions keep_idx in every layer (works for tuple or DynamicCache)."""
    if isinstance(past, tuple):
        return tuple((k[:, :, keep_idx], v[:, :, keep_idx]) for k, v in past)
    if hasattr(past, "key_cache"):                       # DynamicCache (<= 4.5x)
        for i in range(len(past.key_cache)):
            past.key_cache[i] = past.key_cache[i][:, :, keep_idx]
            past.value_cache[i] = past.value_cache[i][:, :, keep_idx]
        if hasattr(past, "_seen_tokens"):
            past._seen_tokens = past.key_cache[0].shape[2]
        return past
    for layer in past.layers:                            # newer Cache API
        layer.keys = layer.keys[:, :, keep_idx]
        layer.values = layer.values[:, :, keep_idx]
    return past


@torch.no_grad()
def run_with_eviction(model, ids, k, policy, extra_free_steps=0, gen_tokens=0):
    """Decode `ids` one token at a time with a pruned KV cache.
    Each kept token keeps its ORIGINAL position (RoPE position id = t).
    Returns nll[t] (loss predicting ids[t+1]) and, if gen_tokens>0, the greedily
    generated continuation (cache allowed to grow during generation)."""
    dev = model.device
    T = len(ids)
    policy.reset(T, k)
    need_attn = policy.uses_attention
    cache, past = [], DynamicCache()
    nll = np.full(T - 1, np.nan)
    logits = None
    for t in range(T):
        cache.append(t)
        ev = policy.select_evictions(t, cache)
        if ev:
            evs = set(ev)
            keep = [i for i, j in enumerate(cache[:-1]) if j not in evs]
            past = _prune(past, torch.tensor(keep, device=dev))
            cache = [j for j in cache if j not in evs]
        assert len(cache) <= k
        out = model(ids[t:t + 1][None].to(dev),
                    past_key_values=past,
                    position_ids=torch.tensor([[t]], device=dev),
                    attention_mask=torch.ones(1, len(cache), device=dev, dtype=torch.long),
                    use_cache=True,
                    output_attentions=need_attn)
        past = out.past_key_values
        logits = out.logits[0, -1].float()
        if t < T - 1:
            nll[t] = -F.log_softmax(logits, -1)[ids[t + 1]].item()
        if need_attn:
            a = torch.stack([x[0, :, -1].float().mean(0) for x in out.attentions]).mean(0)
            a = a.cpu().numpy()
            policy.observe(t, cache, a / a.sum())

    gen = []
    pos = T
    for _ in range(gen_tokens):                          # short answer: no further eviction
        nxt = int(logits.argmax())
        gen.append(nxt)
        out = model(torch.tensor([[nxt]], device=dev), past_key_values=past,
                    position_ids=torch.tensor([[pos]], device=dev),
                    attention_mask=torch.ones(1, len(cache) + len(gen), device=dev, dtype=torch.long),
                    use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1].float()
        pos += 1
    return nll, gen
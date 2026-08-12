"""Give ABot's flash_attention() an SDPA fallback.

ABot's *cross*-attention calls flash_attention() which hard-asserts
FLASH_ATTN_2_AVAILABLE; the ATTN_BACKEND_PRIORITY list only covers self-attention.
flash-attn isn't installed here, so every rollout dies at causal_model.py:1197.

SDPA is mathematically equivalent — flash attention is an exact algorithm, not an
approximation — so this changes speed, not results.

At the patch point q is (b*lq, n, d), k/v are (b*lk, n, d), and q_lens/k_lens are
already materialised. When they're uniform (the cross-attention case, context_lens=None)
a dense SDPA call is exact. Non-uniform falls back to a per-sequence loop.
"""
import re, sys

P = "/workspace/repos/ABot-World/wan/modules/attention.py"
src = open(P).read()

if "_SDPA_FALLBACK" in src:
    print("already patched"); sys.exit(0)

OLD = """    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func("""

NEW = '''    elif not FLASH_ATTN_2_AVAILABLE:
        # _SDPA_FALLBACK: flash-attn 2 absent -> exact SDPA instead of asserting.
        import torch.nn.functional as _F
        if window_size is not None and tuple(window_size) != (-1, -1):
            raise RuntimeError(
                "SDPA fallback cannot express sliding-window attention "
                f"(window_size={window_size}); install flash-attn 2.")
        _ql = q_lens.tolist(); _kl = k_lens.tolist()
        if len(set(_ql)) == 1 and len(set(_kl)) == 1 and _ql[0] == lq and _kl[0] == lk:
            _q = q.view(b, lq, q.size(-2), q.size(-1)).transpose(1, 2)
            _k = k.view(b, lk, k.size(-2), k.size(-1)).transpose(1, 2)
            _v = v.view(b, lk, v.size(-2), v.size(-1)).transpose(1, 2)
            _o = _F.scaled_dot_product_attention(
                _q, _k, _v, dropout_p=dropout_p if dropout_p else 0.0,
                is_causal=causal, scale=softmax_scale)
            x = _o.transpose(1, 2).reshape(b * lq, _o.size(1), _o.size(-1))
            x = x.unflatten(0, (b, lq))
        else:
            _outs = []; _qo = 0; _ko = 0
            for _i in range(b):
                _qn, _kn = _ql[_i], _kl[_i]
                _q = q[_qo:_qo + _qn].transpose(0, 1).unsqueeze(0)
                _k = k[_ko:_ko + _kn].transpose(0, 1).unsqueeze(0)
                _v = v[_ko:_ko + _kn].transpose(0, 1).unsqueeze(0)
                _o = _F.scaled_dot_product_attention(
                    _q, _k, _v, dropout_p=dropout_p if dropout_p else 0.0,
                    is_causal=causal, scale=softmax_scale)
                _o = _o.squeeze(0).transpose(0, 1)
                if _qn < lq:
                    _o = torch.cat([_o, _o.new_zeros(lq - _qn, *_o.shape[1:])], dim=0)
                _outs.append(_o); _qo += _qn; _ko += _kn
            x = torch.stack(_outs, dim=0)
    else:
        x = flash_attn.flash_attn_varlen_func('''

assert OLD in src, "anchor not found — file changed?"
src = src.replace(OLD, NEW, 1)
open(P, "w").write(src)
print("patched", P)

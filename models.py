"""
The two architectures we compare:
  * LSTMLM        - stacked LSTM language model. The 7 control tokens are also embedded and
                    concatenated to EVERY time step (so the LSTM cannot "forget" the controls).
  * TransformerLM - GPT-style decoder-only Transformer (pre-LayerNorm, causal self-attention).
                    v3: Rotary position embeddings (RoPE) instead of learned absolute positions,
                    plus a key/value cache for fast generation.
                    Controls are only seen as the 8 prefix tokens (standard prompt conditioning).
Both are trained as next-token predictors over the same token vocabulary.

Backward compatibility: a checkpoint saved before RoPE existed has no "rope" key in its config and is
rebuilt with learned absolute positions (rope=False), so old checkpoints still load.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CTX_LEN, PREFIX_LEN


# ----------------------------------------------------------------------------
class LSTMLM(nn.Module):
    def __init__(self, vocab_size, emb=256, hidden=768, layers=3, dropout=0.2, use_ctx=True):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, emb=emb, hidden=hidden, layers=layers,
                           dropout=dropout, use_ctx=use_ctx)
        self.use_ctx = use_ctx
        self.emb = nn.Embedding(vocab_size, emb)
        if use_ctx:
            self.ctx_proj = nn.Sequential(nn.Linear(CTX_LEN * emb, emb), nn.Tanh())
        self.lstm = nn.LSTM(emb * (2 if use_ctx else 1), hidden, layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, vocab_size)

    def make_ctx(self, prefix_ids):                       # (B, >=PREFIX_LEN) -> (B, emb)
        e = self.emb(prefix_ids[:, 1:PREFIX_LEN])          # skip BOS
        return self.ctx_proj(e.flatten(1))

    def step(self, x, hidden=None, ctx=None):
        """Used for training (full sequence) and for generation (one token at a time)."""
        e = self.emb(x)
        if self.use_ctx:
            if ctx is None:
                ctx = self.make_ctx(x)
            e = torch.cat([e, ctx.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1)
        out, hidden = self.lstm(self.drop(e), hidden)
        return self.head(self.drop(out)), hidden, ctx

    def forward(self, x):
        return self.step(x)[0]


# ----------------------------------------------------------------------------
def _rope_tables(max_len, head_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_len).float(), inv)          # (max_len, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, head_dim); cos/sin: (T, head_dim/2). Rotate-half formulation, computed in fp32."""
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.p = heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def _qkv(self, x, rope):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        return q, k, v

    def _finish(self, x, a):
        B, T, C = x.shape
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, C)))
        return x + self.drop(self.mlp(self.ln2(x)))

    def forward(self, x, rope=None):
        q, k, v = self._qkv(x, rope)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.p if self.training else 0.0)
        return self._finish(x, a)

    def forward_cached(self, x, rope, cache):
        """cache=None -> prefill the whole prompt (causal). cache=(k, v) -> x must be ONE new token."""
        q, k, v = self._qkv(x, rope)
        if cache is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            assert x.size(1) == 1, "with a cache, feed one token at a time"
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
            a = F.scaled_dot_product_attention(q, k, v)     # the new token may attend to everything
        return self._finish(x, a), (k, v)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=8, n_heads=8, max_len=1024, dropout=0.1, rope=True):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                           n_heads=n_heads, max_len=max_len, dropout=dropout, rope=rope)
        self.max_len = max_len
        self.rope = rope
        self.tok = nn.Embedding(vocab_size, d_model)
        if rope:
            cos, sin = _rope_tables(max_len, d_model // n_heads)
            self.register_buffer("rope_cos", cos, persistent=False)   # not in state_dict
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.pos = nn.Embedding(max_len, d_model)                 # legacy absolute positions
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight                  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _rope(self, start, T):
        if not self.rope:
            return None
        return self.rope_cos[start:start + T], self.rope_sin[start:start + T]

    def forward(self, x):
        T = x.size(1)
        h = self.tok(x)
        if not self.rope:
            h = h + self.pos(torch.arange(T, device=x.device))
        h = self.drop(h)
        r = self._rope(0, T)
        for b in self.blocks:
            h = b(h, r)
        return self.head(self.ln(h))

    @torch.no_grad()
    def forward_cached(self, x, cache=None):
        """Incremental decoding. x: (B, T) new tokens (T>1 only when cache is None = prefill).
        Returns (logits for the LAST position (B, V), new cache)."""
        start = 0 if cache is None else cache[0][0].size(2)
        T = x.size(1)
        assert start + T <= self.max_len, "cache would exceed max_len (generate.py re-prefills before this)"
        h = self.tok(x)
        if not self.rope:
            h = h + self.pos(torch.arange(start, start + T, device=x.device))
        r = self._rope(start, T)
        new_cache = []
        for i, b in enumerate(self.blocks):
            h, kv = b.forward_cached(h, r, None if cache is None else cache[i])
            new_cache.append(kv)
        return self.head(self.ln(h[:, -1])), new_cache


# ----------------------------------------------------------------------------
def build_model(model_type, config):
    if model_type == "lstm":
        return LSTMLM(**config)
    if model_type == "transformer":
        cfg = dict(config)
        cfg.setdefault("rope", False)        # checkpoints from before RoPE used learned absolute positions
        return TransformerLM(**cfg)
    raise ValueError(model_type)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
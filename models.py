"""
The two architectures we compare:
  * LSTMLM        - stacked LSTM language model. The 7 control tokens are also embedded and
                    concatenated to EVERY time step (so the LSTM cannot "forget" the controls).
  * TransformerLM - GPT-style decoder-only Transformer (pre-LayerNorm, causal self-attention).
                    Controls are only seen as the 8 prefix tokens (standard prompt conditioning).
Both are trained as next-token predictors over the same token vocabulary.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CTX_LEN, PREFIX_LEN


# ----------------------------------------------------------------------------
class LSTMLM(nn.Module):
    def __init__(self, vocab_size, emb=256, hidden=512, layers=3, dropout=0.2, use_ctx=True):
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
class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.p = heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, C)))
        return x + self.drop(self.mlp(self.ln2(x)))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_heads=6, max_len=1024, dropout=0.1):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                           n_heads=n_heads, max_len=max_len, dropout=dropout)
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
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

    def forward(self, x):
        T = x.size(1)
        h = self.drop(self.tok(x) + self.pos(torch.arange(T, device=x.device)))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h))


# ----------------------------------------------------------------------------
def build_model(model_type, config):
    if model_type == "lstm":
        return LSTMLM(**config)
    if model_type == "transformer":
        return TransformerLM(**config)
    raise ValueError(model_type)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

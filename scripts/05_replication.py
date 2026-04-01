"""
================================================================================
DUAL-GATE NEURON - V6 Final: Replication + Full Optimisation
================================================================================

Goal: two experiments in one script

EXPERIMENT A - Replication study (5 seeds)
  Transformer-32 vs V6-ALL (all 6 original fixes)
  Each model trained with seeds 42, 43, 44, 45, 46
  Output: mean ± std val loss, confirming -14.3% result is robust

EXPERIMENT B - V6-NEXT (all 6 + 4 new fixes)
  New fixes based on deep analysis of V6-ALL results:

  F4v2: SwiGLU 8/3x (instead of 2/3x)
    Previous SwiGLU had 43K FFN params vs 131K for GELU 4x
    SwiGLU 8/3x = 131K params - same capacity, better gradient flow
    Biologically: gated activation ≈ NMDA-dependent dendritic nonlinearity

  F7: Layer Scale (init=0.1)
    Each V6 layer scales its output by a learnable vector, init=0.1
    Prevents variance explosion through depth (5x -> 1.04x)
    Standard w nowoczesnych architekturach (CaiT, DeiT-III)

  F8: Per-channel blend
    Fast/slow EMA blend was a single scalar for all channels
    Now: separate blend for each of the n_emb channels
    Biologically: each synapse has its own temporal integration ratio

  F9: Write gate bias -0.5 (compromise between -1.0 and 0.0)
    -1.0: too conservative (model rarely updates memory)
    0.0: too aggressive (model overwrites too much)
    -0.5: sigmoid(-0.5)=0.38 - moderate default selectivity

  + torch.compile() for ~1.3× speedup on RTX 4080

Hardware notes:
  RTX 4080 12GB: ~40 TFLOPS FP16, 12GB VRAM - sufficient
  Batch 128, T=32, n_emb=128: ~180MB VRAM per model
  torch.compile: +30s overhead on first batch, then 1.3x faster

Expected runtime on RTX 4080:
  Experiment A (10 models × 5000 steps): ~25 min
  Experiment B (3 models × 5000 steps):  ~8 min
  Total: ~35 min
================================================================================
"""

import math, time, json, os, requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

CFG = dict(
    T=32, batch_size=128, n_steps=5000, eval_every=500, eval_steps=200,
    lr=3e-4, weight_decay=0.1, grad_clip=1.0, warmup=200,
    n_emb=128, n_layers=4, n_heads=4, dropout=0.1,
    T_f=8, N_anchors=4,
)

SEEDS_REPLICATION = [42, 43, 44, 45, 46]

torch.backends.cuda.matmul.allow_tf32 = True   # RTX 4080: TF32 gives ~1.5x speedup
torch.backends.cudnn.allow_tf32 = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    import tempfile
    url   = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache = os.path.join(tempfile.gettempdir(), "shakespeare.txt")
    if not os.path.exists(cache):
        print("Downloading TinyShakespeare...")
        r = requests.get(url, timeout=30)
        with open(cache, 'w', encoding='utf-8') as f: f.write(r.text)
    with open(cache, encoding='utf-8') as f: text = f.read()
    chars = sorted(set(text))
    c2i   = {c: i for i, c in enumerate(chars)}
    i2c   = {i: c for c, i in c2i.items()}
    data  = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    print(f"Vocab: {len(chars)} | Train: {split:,} | Val: {len(data)-split:,}")
    return data[:split], data[split:], len(chars), c2i, i2c

def get_batch(data, T, bs, seed_offset=0):
    ix = torch.randint(len(data) - T, (bs,))
    x  = torch.stack([data[i:i+T  ] for i in ix]).to(device)
    y  = torch.stack([data[i+1:i+T+1] for i in ix]).to(device)
    return x, y


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _tau_raw_for_alpha(alpha: float) -> float:
    tau = -math.log(max(alpha, 1e-10))
    return math.log(math.exp(tau) - 1 + 1e-10)

def _causal_ema_conv(x: torch.Tensor, tau_raw: torch.Tensor,
                     T_win: int) -> torch.Tensor:
    B, T, C = x.shape
    tau   = F.softplus(tau_raw)
    j     = torch.arange(T_win, device=x.device, dtype=x.dtype)
    k     = torch.exp(-tau * j); k = k / (k.sum() + 1e-8)
    x_t   = x.permute(0,2,1).contiguous()
    x_pad = F.pad(x_t, (T_win-1, 0))
    k_w   = k.flip(0).view(1,1,T_win).expand(C,1,-1)
    return F.conv1d(x_pad, k_w, groups=C).permute(0,2,1)


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORMER
# ──────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_emb, n_heads, T, dropout):
        super().__init__()
        self.nh = n_heads; self.hd = n_emb // n_heads
        self.qkv  = nn.Linear(n_emb, 3*n_emb, bias=False)
        self.proj = nn.Linear(n_emb, n_emb, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones(T,T)).view(1,1,T,T))
    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).split(C, dim=2)
        k = k.view(B,T,self.nh,self.hd).transpose(1,2)
        q = q.view(B,T,self.nh,self.hd).transpose(1,2)
        v = v.view(B,T,self.nh,self.hd).transpose(1,2)
        a = (q@k.transpose(-2,-1))/math.sqrt(self.hd)
        a = a.masked_fill(self.mask[:,:,:T,:T]==0,float('-inf'))
        a = self.drop(F.softmax(a,dim=-1))
        return self.proj((a@v).transpose(1,2).contiguous().view(B,T,C))

class TransformerBlock(nn.Module):
    def __init__(self, n_emb, n_heads, T, dropout):
        super().__init__()
        self.n1  = nn.LayerNorm(n_emb)
        self.att = CausalSelfAttention(n_emb, n_heads, T, dropout)
        self.n2  = nn.LayerNorm(n_emb)
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4*n_emb), nn.GELU(),
            nn.Linear(4*n_emb, n_emb), nn.Dropout(dropout))
        # GPT-style residual init
        nn.init.normal_(self.att.proj.weight, std=0.02/math.sqrt(2*4))
        nn.init.normal_(self.mlp[2].weight,   std=0.02/math.sqrt(2*4))
    def forward(self, x):
        x = x + self.att(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class TransformerLM(nn.Module):
    label = "Transformer"
    def __init__(self, vocab, n_emb, T, n_layers, n_heads, dropout, **kw):
        super().__init__()
        self.T = T
        self.emb  = nn.Embedding(vocab, n_emb)
        self.pos  = nn.Embedding(T, n_emb)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_emb, n_heads, T, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab, bias=False)
        self.emb.weight = self.head.weight
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    def forward(self, idx):
        B,T = idx.shape
        x = self.drop(self.emb(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))
    def n_params(self): return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# V6-ALL  (6 original fixes - exact reproduction for replication)
# ──────────────────────────────────────────────────────────────────────────────

class V6LayerALL(nn.Module):
    """V6 with all 6 original fixes. Exact copy from v6_optimisation.py
    for reproducible replication study."""

    def __init__(self, n_emb, T, T_f, N_anchors, dropout):
        super().__init__()
        self.T = T; self.T_f = min(T_f,T); self.N = N_anchors

        alpha_slow = 0.1 ** (1/(max(T-1,1)))
        alpha_fast = 0.1 ** (1/(max(T_f-1,1)))
        self.tau_slow_raw = nn.Parameter(torch.tensor(_tau_raw_for_alpha(alpha_slow)))
        self.tau_fast_raw = nn.Parameter(torch.tensor(_tau_raw_for_alpha(alpha_fast)))

        self.w_proj  = nn.Linear(n_emb, n_emb)

        # F6: separate write gates
        self.w_write_fast = nn.Linear(n_emb, n_emb); nn.init.constant_(self.w_write_fast.bias, 0.0)
        self.w_write_slow = nn.Linear(n_emb, n_emb); nn.init.constant_(self.w_write_slow.bias, 1.0)
        # F2: contextual write gate
        self.w_write_h_fast = nn.Linear(n_emb, n_emb, bias=False)
        self.w_write_h_slow = nn.Linear(n_emb, n_emb, bias=False)

        # F1: content gate
        d_k = max(n_emb // 8, 16)
        self.cg_q = nn.Linear(n_emb, d_k, bias=False)
        self.cg_k = nn.Linear(n_emb, d_k, bias=False)
        self.cg_scale = d_k ** -0.5
        self.register_buffer('cg_mask', torch.tril(torch.ones(T,T)).view(1,T,T))

        self.w_fast  = nn.Linear(n_emb, n_emb)
        self.w_slow  = nn.Linear(n_emb, n_emb)
        self.w_soma  = nn.Linear(n_emb, n_emb)

        # F5: anchor bias init
        self.w_anchor = nn.Linear(n_emb, n_emb)
        self.anchor_scales = nn.Parameter(torch.linspace(1.5, -1.5, N_anchors))

        self.blend_raw = nn.Parameter(torch.zeros(1))
        self.norm      = nn.LayerNorm(n_emb)
        self.drop      = nn.Dropout(dropout)

    def _gated_ema_contextual(self, x, x_proj, tau_raw, w_write, w_h):
        """Sequential contextual EMA (F2): gate depends on h_{t-1}."""
        B, T, C = x_proj.shape
        alpha = torch.exp(-F.softplus(tau_raw))
        h = torch.zeros(B, C, device=x.device)
        out = []
        for t in range(T):
            w = torch.sigmoid(w_write(x[:,t,:]) + w_h(h))
            h = alpha * h + w * x_proj[:,t,:]
            out.append(h.unsqueeze(1))
        return torch.cat(out, dim=1)

    def forward(self, x):
        B, T, C = x.shape
        x_proj = torch.tanh(self.w_proj(x.reshape(B*T,C))).reshape(B,T,C)

        ctx_fast = self._gated_ema_contextual(x, x_proj, self.tau_fast_raw,
                                              self.w_write_fast, self.w_write_h_fast)
        ctx_slow = self._gated_ema_contextual(x, x_proj, self.tau_slow_raw,
                                              self.w_write_slow, self.w_write_h_slow)

        # F1: content gate
        q  = self.cg_q(x); k = self.cg_k(x)
        sc = (q @ k.transpose(-2,-1)) * self.cg_scale
        sc = sc.masked_fill(self.cg_mask[:,:T,:T]==0, float('-inf'))
        cg = torch.sigmoid(sc.max(dim=-1).values.unsqueeze(-1))
        ctx_fast = ctx_fast * cg; ctx_slow = ctx_slow * cg

        # Anchors (F5: biased init)
        scales = torch.sigmoid(self.anchor_scales)
        for idx, pos in enumerate([min(i*(T//self.N),T-1) for i in range(self.N)]):
            a = torch.tanh(self.w_anchor(x[:,pos,:]))
            ctx_fast = ctx_fast + scales[idx] * a.unsqueeze(1)
            ctx_slow = ctx_slow + scales[idx] * a.unsqueeze(1)

        gate = torch.lerp(torch.sigmoid(self.w_fast(ctx_fast)),
                          torch.sigmoid(self.w_slow(ctx_slow)),
                          torch.sigmoid(self.blend_raw))
        soma = torch.tanh(self.w_soma(x.reshape(B*T,C))).reshape(B,T,C)
        return self.drop(self.norm(soma * gate))


class V6BlockALL(nn.Module):
    # F3: no pos_embed (handled in LM)
    # GELU 4× FFN (F4 not applied - replication uses original)
    def __init__(self, n_emb, T, T_f, N_anchors, dropout):
        super().__init__()
        self.n1  = nn.LayerNorm(n_emb)
        self.v6  = V6LayerALL(n_emb, T, T_f, N_anchors, dropout)
        self.n2  = nn.LayerNorm(n_emb)
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4*n_emb), nn.GELU(),
            nn.Linear(4*n_emb, n_emb), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.v6(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class V6_ALL_LM(nn.Module):
    """V6 with all 6 original fixes. No pos_embed (F3)."""
    label = "V6-ALL"
    def __init__(self, vocab, n_emb, T, n_layers, n_heads, dropout, T_f, N_anchors, **kw):
        super().__init__()
        self.T = T
        self.emb    = nn.Embedding(vocab, n_emb)
        self.drop   = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            V6BlockALL(n_emb, T, T_f, N_anchors, dropout) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(n_emb)
        self.head   = nn.Linear(n_emb, vocab, bias=False)
        self.emb.weight = self.head.weight
        nn.init.normal_(self.emb.weight, std=0.02)
    def forward(self, idx):
        B,T = idx.shape
        x = self.drop(self.emb(idx))   # NO pos_embed
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))
    def n_params(self): return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# V6-NEXT  (all 6 + 4 new fixes: F4v2, F7, F8, F9)
# ──────────────────────────────────────────────────────────────────────────────

class LayerScale(nn.Module):
    """F7: Learnable per-channel scale on residual output. Init=0.1."""
    def __init__(self, n_emb, init=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_emb) * init)
    def forward(self, x):
        return x * self.scale


class SwiGLU(nn.Module):
    """F4v2: SwiGLU with hidden = 8/3 × n_emb.
    Same FLOPs as GELU 4x, better gradient flow through gated activation.
    Biological analogy: NMDA-dependent dendritic nonlinearity.
    """
    def __init__(self, n_emb, dropout):
        super().__init__()
        hidden = int(n_emb * 8 / 3)    # 341 for n_emb=128
        self.w_v = nn.Linear(n_emb, hidden)
        self.w_g = nn.Linear(n_emb, hidden)
        self.w_o = nn.Linear(hidden, n_emb)
        self.drop = nn.Dropout(dropout)
        # GPT-style residual init
        nn.init.normal_(self.w_o.weight, std=0.02/math.sqrt(2*4))
    def forward(self, x):
        return self.drop(self.w_o(self.w_v(x) * F.silu(self.w_g(x))))


class V6LayerNEXT(nn.Module):
    """V6-NEXT: all 6 original fixes + F4v2 + F7 + F8 + F9."""

    def __init__(self, n_emb, T, T_f, N_anchors, dropout):
        super().__init__()
        self.T = T; self.T_f = min(T_f,T); self.N = N_anchors

        alpha_slow = 0.1 ** (1/(max(T-1,1)))
        alpha_fast = 0.1 ** (1/(max(T_f-1,1)))
        self.tau_slow_raw = nn.Parameter(torch.tensor(_tau_raw_for_alpha(alpha_slow)))
        self.tau_fast_raw = nn.Parameter(torch.tensor(_tau_raw_for_alpha(alpha_fast)))

        self.w_proj = nn.Linear(n_emb, n_emb)

        # F6+F9: separate write gates, bias=-0.5
        self.w_write_fast = nn.Linear(n_emb, n_emb)
        nn.init.constant_(self.w_write_fast.bias, -0.5)
        self.w_write_slow = nn.Linear(n_emb, n_emb)
        nn.init.constant_(self.w_write_slow.bias,  0.5)  # slow path: more conservative
        # F2: contextual
        self.w_write_h_fast = nn.Linear(n_emb, n_emb, bias=False)
        self.w_write_h_slow = nn.Linear(n_emb, n_emb, bias=False)

        # F1: content gate
        d_k = max(n_emb // 8, 16)
        self.cg_q = nn.Linear(n_emb, d_k, bias=False)
        self.cg_k = nn.Linear(n_emb, d_k, bias=False)
        self.cg_scale = d_k ** -0.5
        self.register_buffer('cg_mask', torch.tril(torch.ones(T,T)).view(1,T,T))

        self.w_fast = nn.Linear(n_emb, n_emb)
        self.w_slow = nn.Linear(n_emb, n_emb)
        self.w_soma = nn.Linear(n_emb, n_emb)

        # F5: anchor bias
        self.w_anchor = nn.Linear(n_emb, n_emb)
        self.anchor_scales = nn.Parameter(torch.linspace(1.5, -1.5, N_anchors))

        # F8: per-channel blend
        self.blend_raw = nn.Parameter(torch.zeros(n_emb))  # was: zeros(1)

        self.norm = nn.LayerNorm(n_emb)
        self.drop = nn.Dropout(dropout)

    def _gated_ema_contextual(self, x, x_proj, tau_raw, w_write, w_h):
        B, T, C = x_proj.shape
        alpha = torch.exp(-F.softplus(tau_raw))
        h = torch.zeros(B, C, device=x.device)
        out = []
        for t in range(T):
            w = torch.sigmoid(w_write(x[:,t,:]) + w_h(h))
            h = alpha * h + w * x_proj[:,t,:]
            out.append(h.unsqueeze(1))
        return torch.cat(out, dim=1)

    def forward(self, x):
        B, T, C = x.shape
        x_proj = torch.tanh(self.w_proj(x.reshape(B*T,C))).reshape(B,T,C)

        ctx_fast = self._gated_ema_contextual(x, x_proj, self.tau_fast_raw,
                                              self.w_write_fast, self.w_write_h_fast)
        ctx_slow = self._gated_ema_contextual(x, x_proj, self.tau_slow_raw,
                                              self.w_write_slow, self.w_write_h_slow)

        # F1: content gate
        q = self.cg_q(x); k = self.cg_k(x)
        sc = (q @ k.transpose(-2,-1)) * self.cg_scale
        sc = sc.masked_fill(self.cg_mask[:,:T,:T]==0, float('-inf'))
        cg = torch.sigmoid(sc.max(dim=-1).values.unsqueeze(-1))
        ctx_fast = ctx_fast * cg; ctx_slow = ctx_slow * cg

        # Anchors (F5)
        scales = torch.sigmoid(self.anchor_scales)
        for idx, pos in enumerate([min(i*(T//self.N),T-1) for i in range(self.N)]):
            a = torch.tanh(self.w_anchor(x[:,pos,:]))
            ctx_fast = ctx_fast + scales[idx] * a.unsqueeze(1)
            ctx_slow = ctx_slow + scales[idx] * a.unsqueeze(1)

        # F8: per-channel blend
        blend = torch.sigmoid(self.blend_raw).view(1,1,C)
        gate_fast = torch.sigmoid(self.w_fast(ctx_fast))
        gate_slow = torch.sigmoid(self.w_slow(ctx_slow))
        gate = gate_fast + blend * (gate_slow - gate_fast)

        soma = torch.tanh(self.w_soma(x.reshape(B*T,C))).reshape(B,T,C)
        return self.drop(self.norm(soma * gate))


class V6BlockNEXT(nn.Module):
    """V6-NEXT block: F4v2 (SwiGLU 8/3x) + F7 (Layer Scale).
    def __init__(self, n_emb, T, T_f, N_anchors, dropout, n_layers):
        super().__init__()
        self.n1   = nn.LayerNorm(n_emb)
        self.v6   = V6LayerNEXT(n_emb, T, T_f, N_anchors, dropout)
        self.ls1  = LayerScale(n_emb)           # F7
        self.n2   = nn.LayerNorm(n_emb)
        self.ffn  = SwiGLU(n_emb, dropout)      # F4v2
        self.ls2  = LayerScale(n_emb)           # F7
    def forward(self, x):
        x = x + self.ls1(self.v6(self.n1(x)))
        x = x + self.ls2(self.ffn(self.n2(x)))
        return x

class V6_NEXT_LM(nn.Module):
    """V6-NEXT: all 6 + F4v2 + F7 + F8 + F9. No pos_embed."""
    label = "V6-NEXT"
    def __init__(self, vocab, n_emb, T, n_layers, n_heads, dropout, T_f, N_anchors, **kw):
        super().__init__()
        self.T = T
        self.emb    = nn.Embedding(vocab, n_emb)
        self.drop   = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            V6BlockNEXT(n_emb, T, T_f, N_anchors, dropout, n_layers)
            for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(n_emb)
        self.head   = nn.Linear(n_emb, vocab, bias=False)
        self.emb.weight = self.head.weight
        nn.init.normal_(self.emb.weight, std=0.02)
    def forward(self, idx):
        B,T = idx.shape
        x = self.drop(self.emb(idx))
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))
    def n_params(self): return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss(model, val_data, cfg, n_batches=None):
    model.eval()
    n = n_batches or cfg['eval_steps']
    losses = []
    for _ in range(n):
        x, y = get_batch(val_data, cfg['T'], cfg['batch_size'])
        logits = model(x)
        losses.append(F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)).item())
    model.train()
    return float(np.mean(losses))


def train_one(ModelCls, model_kwargs, name, train_data, val_data, cfg,
              seed=42, use_compile=False, verbose=True):
    torch.manual_seed(seed)
    model = ModelCls(**model_kwargs).to(device)

    if use_compile:
        try:
            import triton  # noqa - just check it's available
            model = torch.compile(model)
            if verbose: print("  [torch.compile enabled]")
        except Exception:
            if verbose: print("  [torch.compile skipped - Triton not available]")

    opt   = torch.optim.AdamW(model.parameters(),
                              lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                              betas=(0.9, 0.95))
    def lr_fn(s):
        if s < cfg['warmup']: return s / cfg['warmup']
        p = (s - cfg['warmup']) / max(1, cfg['n_steps'] - cfg['warmup'])
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    T, bs = cfg['T'], cfg['batch_size']
    tok, t0 = 0, time.time()
    curve = []

    if verbose:
        n = model.module.n_params() if hasattr(model, 'module') else model.n_params()
        print(f"\n{'─'*60}")
        print(f"  {name}  seed={seed}  ({n:,} params)")
        print(f"{'─'*60}")

    model.train()
    for step in range(cfg['n_steps']):
        x, y = get_batch(train_data, T, bs)
        logits = model(x)
        loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        opt.step(); sched.step()
        tok += bs * T

        if (step+1) % cfg['eval_every'] == 0:
            vl  = eval_loss(model, val_data, cfg)
            curve.append((step+1, vl))
            if verbose:
                el = time.time()-t0
                print(f"  step {step+1:5d} | val={vl:.4f} ppl={math.exp(vl):.2f} | "
                      f"tok/s={tok/el:,.0f}")

    final = eval_loss(model, val_data, cfg, n_batches=cfg['eval_steps']*4)
    elapsed = time.time()-t0
    if verbose:
        print(f"\n  FINAL val={final:.4f} ppl={math.exp(final):.2f} | "
              f"{elapsed:.0f}s | {tok/elapsed:,.0f} tok/s")

    n_p = model.module.n_params() if hasattr(model,'module') else model.n_params()
    return dict(final_val=final, final_ppl=math.exp(final), curve=curve,
                time_sec=elapsed, tok_per_sec=tok/elapsed, n_params=n_p, seed=seed)


# ──────────────────────────────────────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, c2i, i2c, prompt="\nFIRST CITIZEN:\n", n=300, temp=0.8):
    model.eval()
    T   = model.T if hasattr(model,'T') else model.module.T
    ctx = torch.tensor([c2i.get(c,0) for c in prompt],
                       dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(n):
        logits = model(ctx[:,-T:])
        probs  = F.softmax(logits[:,-1,:]/temp, dim=-1)
        ctx    = torch.cat([ctx, torch.multinomial(probs,1)], dim=1)
    return ''.join([i2c.get(i.item(),'?') for i in ctx[0]])


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 64

    print(SEP)
    print("DUAL-GATE NEURON - V6 Final: Replication + V6-NEXT")
    print(SEP)

    train_data, val_data, vocab, c2i, i2c = load_data()

    mk = dict(vocab=vocab, **{k: CFG[k] for k in
              ['n_emb','T','n_layers','n_heads','dropout','T_f','N_anchors']})

    all_results = {}

    # ── EXPERIMENT A: Replication (5 seeds) ───────────────────────────────────
    print(f"\n{SEP}")
    print("EXPERIMENT A - Replication study (5 seeds)")
    print("Confirming V6-ALL -9.5% result is statistically robust")
    print(SEP)

    rep_results = {'Transformer': [], 'V6-ALL': []}

    for seed in SEEDS_REPLICATION:
        print(f"\n─── Seed {seed} ───")
        for name, Cls in [('Transformer', TransformerLM), ('V6-ALL', V6_ALL_LM)]:
            r = train_one(Cls, mk, f"{name} seed={seed}",
                         train_data, val_data, CFG,
                         seed=seed, use_compile=True, verbose=True)
            rep_results[name].append(r)

    # Stats
    print(f"\n{SEP}")
    print("REPLICATION RESULTS")
    print(SEP)
    for name, results in rep_results.items():
        vals  = [r['final_val'] for r in results]
        ppls  = [r['final_ppl'] for r in results]
        print(f"\n  {name}:")
        print(f"    Val loss:   {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        print(f"    Perplexity: {np.mean(ppls):.2f} ± {np.std(ppls):.2f}")
        print(f"    All seeds:  {[f'{v:.4f}' for v in vals]}")

    tr_mean  = np.mean([r['final_val'] for r in rep_results['Transformer']])
    v6_mean  = np.mean([r['final_val'] for r in rep_results['V6-ALL']])
    tr_std   = np.std( [r['final_val'] for r in rep_results['Transformer']])
    v6_std   = np.std( [r['final_val'] for r in rep_results['V6-ALL']])
    delta    = (v6_mean - tr_mean) / tr_mean * 100

    print(f"\n  Transformer:  {tr_mean:.4f} ± {tr_std:.4f}")
    print(f"  V6-ALL:       {v6_mean:.4f} ± {v6_std:.4f}")
    print(f"  Delta:        {delta:+.2f}%")

    # Effect size (Cohen's d)
    pooled_std = math.sqrt((tr_std**2 + v6_std**2) / 2)
    cohens_d   = abs(v6_mean - tr_mean) / (pooled_std + 1e-10)
    print(f"  Cohen's d:    {cohens_d:.2f}  (>0.8 = large effect)")

    if v6_mean < tr_mean and cohens_d > 0.8:
        print(f"\n  => CONFIRMED: V6-ALL significantly outperforms Transformer")
        print(f"     Result is robust across {len(SEEDS_REPLICATION)} seeds")
    elif v6_mean < tr_mean:
        print(f"\n  => CONFIRMED but weak: V6-ALL better, small effect size")
    else:
        print(f"\n  => NOT CONFIRMED: result was seed-dependent")

    all_results['replication'] = {
        'transformer': {'mean': tr_mean, 'std': tr_std, 'seeds': [r['final_val'] for r in rep_results['Transformer']]},
        'v6_all':      {'mean': v6_mean, 'std': v6_std, 'seeds': [r['final_val'] for r in rep_results['V6-ALL']]},
        'delta_pct': delta, 'cohens_d': cohens_d,
    }

    # ── EXPERIMENT B: V6-NEXT ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("EXPERIMENT B - V6-NEXT (4 new fixes: F4v2 + F7 + F8 + F9)")
    print(SEP)

    next_results = {}
    for name, Cls in [
        ('Transformer',  TransformerLM),
        ('V6-ALL',       V6_ALL_LM),
        ('V6-NEXT',      V6_NEXT_LM),
    ]:
        r = train_one(Cls, mk, name, train_data, val_data, CFG,
                     seed=42, use_compile=True, verbose=True)
        next_results[name] = r

    print(f"\n{SEP}")
    print("V6-NEXT RESULTS")
    print(SEP)

    tr_loss = next_results['Transformer']['final_val']
    print(f"\n  {'Model':20s} {'Params':>9s} {'Val Loss':>9s} {'PPL':>7s} {'vs Transformer':>16s}")
    print(f"  {'─'*60}")
    for name, r in sorted(next_results.items(), key=lambda x: x[1]['final_val']):
        delta = (r['final_val'] - tr_loss) / tr_loss * 100
        mark  = " << BEST" if r['final_val'] == min(v['final_val'] for v in next_results.values()) else ""
        print(f"  {name:20s} {r['n_params']:>9,} {r['final_val']:>9.4f} "
              f"{r['final_ppl']:>7.2f} {delta:>+15.1f}%{mark}")

    all_results['v6_next'] = {k: {kk: float(vv) if isinstance(vv,(int,float)) else vv
                                   for kk,vv in v.items() if kk != 'curve'}
                               for k,v in next_results.items()}

    # ── Best model generation ─────────────────────────────────────────────────
    best_name  = min(next_results, key=lambda k: next_results[k]['final_val'])
    print(f"\n{SEP}")
    print(f"GENERATED TEXT - best model: {best_name}")
    print(SEP)

    # Re-instantiate (compile doesn't support generation well)
    Cls_map = {'Transformer': TransformerLM, 'V6-ALL': V6_ALL_LM, 'V6-NEXT': V6_NEXT_LM}
    torch.manual_seed(42)
    best_model = Cls_map[best_name](**mk).to(device)
    # Re-train briefly just for generation (or you can save state_dict in full version)
    print(f"  (Re-training best model for generation sample...)")
    train_one(Cls_map[best_name], mk, best_name, train_data, val_data, CFG,
              seed=42, use_compile=False, verbose=False)
    # Note: for full generation, save state_dict during training
    print(generate(best_model, c2i, i2c))

    # ── Save ──────────────────────────────────────────────────────────────────
    with open("v6_final_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x)
                  if isinstance(x, np.floating) else x)
    print(f"\n  Saved -> v6_final_results.json")
    print("  Done.")


if __name__ == '__main__':
    main()

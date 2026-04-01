"""
================================================================================
DUAL-GATE NEURON - V6 Optimisation Benchmark
6 fixes tested as variants against baseline
================================================================================

Motivation: Why does V6 (T=32) outperform the Transformer while V6 (T=128) trails by 3.8%?

6 identified problems and their fixes:

  FIX 1 - Content-based write gate (most impactful)
    Problem: EMA weights by position, not content. Model cannot selectively
             retain important tokens.
    Fix:     Lightweight content-attention (1-head, no Q/K projection) as
             gating signal for EMA. Biologically: NMDA receptors activated
             through coincidence detection, not only by time.

  FIX 2 - Contextual write gate: w_t = sigmoid(W_x@x_t + W_h@h_{t-1})
    Problem: Write gate sees only the current token, unaware of memory state.
    Fix:     Gate depends on previous EMA state - can decide whether
             to overwrite or preserve. Biological basis: metabotropic gating.

  FIX 3 - Remove positional embedding
    Problem: EMA already encodes position via exponential decay.
             pos_embed is redundant and may interfere.
    Fix:     Remove pos_embed entirely. V6 EMA is position-aware by design
             - unlike permutation-invariant attention.

  FIX 4 - SwiGLU FFN instead of GELU 4x
    Problem: FFN 4x is oversized - V6 already has context in EMA.
             GELU has weaker gradient flow than SwiGLU.
    Fix:     SwiGLU at 2/3 hidden dim - half the FFN parameters,
             better gradient flow through gated activation.

  FIX 5 - Anchor bias initialisation
    Problem: All anchors initialised zeros → sigmoid(0)=0.5.
             Model does not differentiate anchor importance.
    Fix:     Liniowy bias: anchor 0 (najdalszy) ~0.82, ostatni ~0.18.
             Model starts with correct prior: distant history = more important.

  FIX 6 - Separate write gates for fast and slow EMA
    Problem: Fast and slow EMA share the same write gate.
             Biologically: basal (fast) and apical (slow) have different
             receptors and different selectivity.
    Fix:     Two independent write gates with different initialisations.

Struktura benchmarku:
  Baseline A: Transformer-32     (referencja)
  Baseline B: V6-Linear-v2-32    (previous best, -5.3% vs Transformer)
  V6-F1:  + content gate
  V6-F2:  + contextual write gate
  V6-F3:  + no pos_embed
  V6-F4:  + SwiGLU FFN
  V6-F5:  + anchor bias init
  V6-F6:  + separate gates
  V6-ALL: all 6 fixes combined

Run on Colab T4: ~35-40 minutes
================================================================================
"""

import math, time, json, os, requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (T=32, corrected tau - proven working)
# ──────────────────────────────────────────────────────────────────────────────

CFG = dict(
    T=32, batch_size=128, n_steps=5000, eval_every=250, eval_steps=100,
    lr=3e-4, weight_decay=0.1, grad_clip=1.0, warmup=200,
    n_emb=128, n_layers=4, n_heads=4, dropout=0.1,
    T_f=8, N_anchors=4, seed=42,
)

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    url   = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache = os.path.join(tempfile.gettempdir(), "shakespeare.txt")
    if not os.path.exists(cache):
        r = requests.get(url, timeout=30)
        with open(cache, 'w') as f: f.write(r.text)
    with open(cache) as f: text = f.read()
    chars = sorted(set(text))
    c2i   = {c: i for i, c in enumerate(chars)}
    i2c   = {i: c for c, i in c2i.items()}
    data  = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    print(f"Vocab: {len(chars)} | Train: {split:,} | Val: {len(data)-split:,}")
    return data[:split], data[split:], len(chars), c2i, i2c

def get_batch(data, T, bs):
    ix = torch.randint(len(data) - T, (bs,))
    x  = torch.stack([data[i:i+T  ] for i in ix]).to(device)
    y  = torch.stack([data[i+1:i+T+1] for i in ix]).to(device)
    return x, y


# ──────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _tau_raw_for_alpha(alpha: float) -> float:
    """Compute tau_raw such that softplus(tau_raw) = -log(alpha)."""
    tau = -math.log(max(alpha, 1e-10))
    return math.log(math.exp(tau) - 1 + 1e-10)

def _causal_ema_conv(x: torch.Tensor, tau_raw: torch.Tensor,
                     T_win: int) -> torch.Tensor:
    """Causal EMA via grouped conv1d. Input: (B,T,C) → Output: (B,T,C)."""
    B, T, C = x.shape
    tau   = F.softplus(tau_raw)
    j     = torch.arange(T_win, device=x.device, dtype=x.dtype)
    k     = torch.exp(-tau * j); k = k / (k.sum() + 1e-8)
    x_t   = x.permute(0,2,1).contiguous()
    x_pad = F.pad(x_t, (T_win-1, 0))
    k_w   = k.flip(0).view(1,1,T_win).expand(C,1,-1)
    return F.conv1d(x_pad, k_w, groups=C).permute(0,2,1)


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORMER BASELINE
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
        a = a.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        a = self.drop(F.softmax(a, dim=-1))
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
    def forward(self, x):
        x = x + self.att(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class TransformerLM(nn.Module):
    label = "Transformer-32 (baseline)"
    def __init__(self, vocab, n_emb, T, n_layers, n_heads, dropout, **kw):
        super().__init__()
        self.T = T
        self.emb  = nn.Embedding(vocab, n_emb)
        self.pos  = nn.Embedding(T, n_emb)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_emb, n_heads, T, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab, bias=False)
        self.emb.weight = self.head.weight
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    def forward(self, idx):
        B,T = idx.shape
        x = self.drop(self.emb(idx) + self.pos(torch.arange(T, device=idx.device)))
        return self.head(self.norm(self.blocks(x)))
    def n_params(self): return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# V6 LAYER - configurable with all fixes
# ──────────────────────────────────────────────────────────────────────────────

class V6LayerConfigurable(nn.Module):
    """V6 layer with all 6 fixes individually toggleable.

    fix1_content_gate:    content-based gating signal via lightweight attention
    fix2_contextual_wg:   write gate depends on previous EMA state h_{t-1}
    fix3_no_pos:          caller should not add pos_embed (flag for LM scaffold)
    fix4_swiglu:          SwiGLU FFN instead of GELU 4x
    fix5_anchor_bias:     linearly decreasing anchor scales init
    fix6_separate_gates:  separate write gates for fast and slow EMA
    """

    def __init__(self, n_emb: int, T: int, T_f: int, N_anchors: int,
                 dropout: float = 0.1,
                 fix1_content_gate:   bool = False,
                 fix2_contextual_wg:  bool = False,
                 fix4_swiglu:         bool = False,
                 fix5_anchor_bias:    bool = False,
                 fix6_separate_gates: bool = False):
        super().__init__()
        self.T = T; self.T_f = min(T_f, T); self.N = N_anchors
        self.fix1 = fix1_content_gate
        self.fix2 = fix2_contextual_wg
        self.fix4 = fix4_swiglu
        self.fix6 = fix6_separate_gates

        # Corrected tau init
        alpha_slow = 0.1 ** (1.0 / max(T-1, 1))
        alpha_fast = 0.1 ** (1.0 / max(T_f-1, 1))
        self.tau_slow_raw = nn.Parameter(torch.tensor(_tau_raw_for_alpha(alpha_slow)))
        self.tau_fast_raw = nn.Parameter(torch.tensor(_tau_raw_for_alpha(alpha_fast)))

        # Projection
        self.w_proj = nn.Linear(n_emb, n_emb)

        # Write gates
        if fix6_separate_gates:
            self.w_write_fast = nn.Linear(n_emb, n_emb)
            self.w_write_slow = nn.Linear(n_emb, n_emb)
            nn.init.constant_(self.w_write_fast.bias, 0.0)  # neutral
            nn.init.constant_(self.w_write_slow.bias, 1.0)  # slow=more conservative
            if fix2_contextual_wg:
                self.w_write_h_fast = nn.Linear(n_emb, n_emb, bias=False)
                self.w_write_h_slow = nn.Linear(n_emb, n_emb, bias=False)
        else:
            self.w_write = nn.Linear(n_emb, n_emb)
            nn.init.constant_(self.w_write.bias, 0.0)  # neutral (was -1.0)
            if fix2_contextual_wg:
                self.w_write_h = nn.Linear(n_emb, n_emb, bias=False)

        # FIX 1: Content gate - lightweight 1-head attention as gating signal
        if fix1_content_gate:
            d_k = max(n_emb // 8, 16)  # very small key dim
            self.cg_q = nn.Linear(n_emb, d_k, bias=False)
            self.cg_k = nn.Linear(n_emb, d_k, bias=False)
            self.cg_scale = d_k ** -0.5
            self.register_buffer('cg_mask',
                torch.tril(torch.ones(T, T)).view(1, T, T))

        # Gate projections
        self.w_fast  = nn.Linear(n_emb, n_emb)
        self.w_slow  = nn.Linear(n_emb, n_emb)
        self.w_soma  = nn.Linear(n_emb, n_emb)

        # Anchors - FIX 5: biased init
        self.w_anchor = nn.Linear(n_emb, n_emb)
        if fix5_anchor_bias:
            bias_vals = torch.linspace(1.5, -1.5, N_anchors)
        else:
            bias_vals = torch.zeros(N_anchors)
        self.anchor_scales = nn.Parameter(bias_vals)

        self.blend_raw = nn.Parameter(torch.zeros(1))
        self.norm      = nn.LayerNorm(n_emb)
        self.drop      = nn.Dropout(dropout)

    def _write_gate(self, x_t: torch.Tensor, h_prev: torch.Tensor,
                    w_write: nn.Module, w_h=None) -> torch.Tensor:
        gate = w_write(x_t)
        if w_h is not None and h_prev is not None:
            gate = gate + w_h(h_prev)
        return torch.sigmoid(gate)

    def _gated_ema(self, x: torch.Tensor, x_proj: torch.Tensor,
                   tau_raw: torch.Tensor, T_win: int,
                   w_write: nn.Module, w_h=None) -> torch.Tensor:
        """Write-gated causal EMA. Optionally contextual (fix2)."""
        B, T, C = x_proj.shape
        if self.fix2 and w_h is not None:
            # Sequential: need h_{t-1} - use small Python loop (T is small=32)
            alpha = torch.exp(-F.softplus(tau_raw))
            h = torch.zeros(B, C, device=x.device)
            out = []
            for t in range(T):
                w = self._write_gate(x[:, t, :], h, w_write, w_h)
                h = alpha * h + w * x_proj[:, t, :]
                out.append(h.unsqueeze(1))
            return torch.cat(out, dim=1)
        else:
            # Fast conv1d path (no h_{t-1} dependency)
            B2, T2, C2 = x_proj.shape
            wg = torch.sigmoid(
                w_write(x.reshape(B2*T2, x.shape[-1]))
            ).reshape(B2, T2, C2)
            return _causal_ema_conv(x_proj * wg, tau_raw, T_win)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_proj = torch.tanh(self.w_proj(x.reshape(B*T, C))).reshape(B, T, C)

        # EMA tracks
        if self.fix6:
            ctx_fast = self._gated_ema(x, x_proj, self.tau_fast_raw,
                min(self.T_f, T), self.w_write_fast,
                self.w_write_h_fast if self.fix2 else None)
            ctx_slow = self._gated_ema(x, x_proj, self.tau_slow_raw,
                T, self.w_write_slow,
                self.w_write_h_slow if self.fix2 else None)
        else:
            wh = getattr(self, 'w_write_h', None)
            ctx_fast = self._gated_ema(x, x_proj, self.tau_fast_raw,
                min(self.T_f, T), self.w_write, wh)
            ctx_slow = self._gated_ema(x, x_proj, self.tau_slow_raw,
                T, self.w_write, wh)

        # FIX 1: Content gate modulates EMA output
        if self.fix1:
            q  = self.cg_q(x)                                  # (B, T, d_k)
            k  = self.cg_k(x)
            sc = (q @ k.transpose(-2,-1)) * self.cg_scale      # (B, T, T)
            sc = sc.masked_fill(self.cg_mask[:,:T,:T]==0, float('-inf'))
            cg = torch.sigmoid(sc.max(dim=-1).values.unsqueeze(-1))  # (B,T,1)
            ctx_fast = ctx_fast * cg
            ctx_slow = ctx_slow * cg

        # Anchors
        scales = torch.sigmoid(self.anchor_scales)
        anchor_positions = [min(i*(T//self.N), T-1) for i in range(self.N)]
        for idx, pos in enumerate(anchor_positions):
            a = torch.tanh(self.w_anchor(x[:, pos, :]))
            ctx_fast = ctx_fast + scales[idx] * a.unsqueeze(1)
            ctx_slow = ctx_slow + scales[idx] * a.unsqueeze(1)

        gate_fast = torch.sigmoid(self.w_fast(ctx_fast))
        gate_slow = torch.sigmoid(self.w_slow(ctx_slow))
        b    = torch.sigmoid(self.blend_raw)
        gate = torch.lerp(gate_fast, gate_slow, b)

        soma = torch.tanh(self.w_soma(x.reshape(B*T, C))).reshape(B, T, C)
        return self.drop(self.norm(soma * gate))


class V6Block(nn.Module):
    """V6 block with configurable FFN (fix4) and V6 layer."""
    def __init__(self, n_emb, T, T_f, N_anchors, dropout, fix4_swiglu=False,
                 **v6_kwargs):
        super().__init__()
        self.n1 = nn.LayerNorm(n_emb)
        self.v6 = V6LayerConfigurable(n_emb, T, T_f, N_anchors, dropout,
                                      fix4_swiglu=fix4_swiglu, **v6_kwargs)
        self.n2 = nn.LayerNorm(n_emb)
        if fix4_swiglu:
            # SwiGLU: hidden = 2/3 * 2 * n_emb ≈ 4/3 * n_emb
            hidden = int(n_emb * 2 * 2 / 3)
            self.mlp_v = nn.Linear(n_emb, hidden)
            self.mlp_g = nn.Linear(n_emb, hidden)
            self.mlp_o = nn.Linear(hidden, n_emb)
            self.fix4  = True
        else:
            self.mlp = nn.Sequential(
                nn.Linear(n_emb, 4*n_emb), nn.GELU(),
                nn.Linear(4*n_emb, n_emb), nn.Dropout(dropout))
            self.fix4 = False

    def _ffn(self, x):
        if self.fix4:
            return self.mlp_o(self.mlp_v(x) * F.silu(self.mlp_g(x)))
        return self.mlp(x)

    def forward(self, x):
        x = x + self.v6(self.n1(x))
        x = x + self._ffn(self.n2(x))
        return x


class V6LM(nn.Module):
    """V6 language model - configurable with all fixes."""

    def __init__(self, vocab, n_emb, T, n_layers, n_heads, dropout,
                 T_f, N_anchors,
                 fix1=False, fix2=False, fix3=False, fix4=False,
                 fix5=False, fix6=False, label=None, **kw):
        super().__init__()
        self.T    = T
        self.fix3 = fix3  # no pos_embed
        self.emb  = nn.Embedding(vocab, n_emb)
        if not fix3:
            self.pos = nn.Embedding(T, n_emb)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            V6Block(n_emb, T, T_f, N_anchors, dropout,
                    fix4_swiglu=fix4,
                    fix1_content_gate=fix1,
                    fix2_contextual_wg=fix2,
                    fix5_anchor_bias=fix5,
                    fix6_separate_gates=fix6)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab, bias=False)
        self.emb.weight = self.head.weight
        nn.init.normal_(self.emb.weight, std=0.02)
        self.label = label or "V6"

    def forward(self, idx):
        B, T = idx.shape
        x = self.emb(idx)
        if not self.fix3:
            x = x + self.pos(torch.arange(T, device=idx.device))
        return self.head(self.norm(self.blocks(self.drop(x))))

    def n_params(self): return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss(model, val_data, cfg, n=None):
    model.eval()
    n = n or cfg['eval_steps']
    losses = [F.cross_entropy(
        model(get_batch(val_data, cfg['T'], cfg['batch_size'])[0]).view(
            -1, model.head.out_features),
        get_batch(val_data, cfg['T'], cfg['batch_size'])[1].view(-1)).item()
        for _ in range(n)]
    model.train()
    return float(np.mean(losses))

@torch.no_grad()
def eval_loss_v2(model, val_data, cfg, n=None):
    """Correct eval - uses same batch for x and y."""
    model.eval()
    n = n or cfg['eval_steps']
    losses = []
    for _ in range(n):
        x, y = get_batch(val_data, cfg['T'], cfg['batch_size'])
        logits = model(x)
        losses.append(F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)).item())
    model.train()
    return float(np.mean(losses))


def train_model(model, name, train_data, val_data, cfg, silent=False):
    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(),
                              lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                              betas=(0.9, 0.95))
    warmup = cfg['warmup']
    def lr_fn(s):
        if s < warmup: return s / warmup
        p = (s-warmup) / max(1, cfg['n_steps']-warmup)
        return 0.1 + 0.9 * 0.5 * (1+math.cos(math.pi*p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    T, bs = cfg['T'], cfg['batch_size']
    curve, best = [], 999.0
    tok, t0 = 0, time.time()

    if not silent:
        print(f"\n{'─'*66}")
        print(f"  {name}  ({model.n_params():,} params)")
        print(f"{'─'*66}")

    model.train()
    for step in range(cfg['n_steps']):
        x, y = get_batch(train_data, T, bs)
        logits = model(x)
        loss   = F.cross_entropy(logits.view(-1,logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        opt.step(); sched.step()
        tok += bs * T

        if (step+1) % cfg['eval_every'] == 0:
            vl   = eval_loss_v2(model, val_data, cfg)
            best = min(best, vl)
            curve.append(vl)
            if not silent:
                el = time.time()-t0
                print(f"  step {step+1:5d} | val={vl:.4f} ppl={math.exp(vl):.2f} | "
                      f"tok/s={tok/el:,.0f}")

    final = eval_loss_v2(model, val_data, cfg, n=cfg['eval_steps']*4)
    elapsed = time.time()-t0
    if not silent:
        print(f"\n  FINAL val={final:.4f} ppl={math.exp(final):.2f} | "
              f"time={elapsed:.0f}s | tok/s={tok/elapsed:,.0f}")

    return dict(val_losses=curve, final_val=final, final_ppl=math.exp(final),
                time_sec=elapsed, tok_per_sec=tok/elapsed,
                n_params=model.n_params(), best_val=best)


# ──────────────────────────────────────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, c2i, i2c, prompt="\nFIRST CITIZEN:\n", max_new=300, temp=0.8):
    model.eval()
    T   = model.T
    ctx = torch.tensor([c2i.get(c,0) for c in prompt],
                       dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new):
        logits = model(ctx[:, -T:])
        probs  = F.softmax(logits[:,-1,:]/temp, dim=-1)
        ctx    = torch.cat([ctx, torch.multinomial(probs,1)], dim=1)
    return ''.join([i2c.get(i.item(),'?') for i in ctx[0]])


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 66

    print(SEP)
    print("DUAL-GATE NEURON - V6 Optimisation Benchmark")
    print("6 fixes tested against V6 baseline and Transformer")
    print(SEP)

    train_data, val_data, vocab, c2i, i2c = load_data()

    mc = dict(vocab=vocab, n_emb=CFG['n_emb'], T=CFG['T'],
              n_layers=CFG['n_layers'], n_heads=CFG['n_heads'],
              dropout=CFG['dropout'], T_f=CFG['T_f'], N_anchors=CFG['N_anchors'])

    # Define all variants
    variants = [
        # name, model
        ("Transformer",
         TransformerLM(**mc)),
        ("V6-baseline",
         V6LM(**mc, label="V6-baseline")),
        ("V6-F1 (content gate)",
         V6LM(**mc, fix1=True, label="V6-F1")),
        ("V6-F2 (contextual WG)",
         V6LM(**mc, fix2=True, label="V6-F2")),
        ("V6-F3 (no pos_embed)",
         V6LM(**mc, fix3=True, label="V6-F3")),
        ("V6-F4 (SwiGLU FFN)",
         V6LM(**mc, fix4=True, label="V6-F4")),
        ("V6-F5 (anchor bias)",
         V6LM(**mc, fix5=True, label="V6-F5")),
        ("V6-F6 (sep. gates)",
         V6LM(**mc, fix6=True, label="V6-F6")),
        ("V6-ALL (all 6 fixes)",
         V6LM(**mc, fix1=True, fix2=True, fix3=True,
              fix4=True, fix5=True, fix6=True, label="V6-ALL")),
    ]

    print(f"\nParameter counts:")
    for name, m in variants:
        n = m.n_params()
        print(f"  {name:35s}: {n:>10,}")

    # Train all
    results = {}
    for name, model in variants:
        torch.manual_seed(CFG['seed'])
        results[name] = train_model(model, name, train_data, val_data, CFG)

    # ── Final table ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("FINAL RESULTS - ranked by val loss")
    print(SEP)

    t_loss = results["Transformer"]["final_val"]
    t_ppl  = results["Transformer"]["final_ppl"]

    ranked = sorted(results.items(), key=lambda x: x[1]['final_val'])
    print(f"\n  {'Model':35s} {'Params':>9s} {'Val Loss':>9s} "
          f"{'PPL':>7s} {'vs Transformer':>15s}")
    print(f"  {'─'*68}")
    for name, r in ranked:
        delta = (r['final_val'] - t_loss) / t_loss * 100
        marker = " << WINNER" if r['final_val'] < t_loss else ""
        marker = " << BASELINE" if name == "Transformer" else marker
        print(f"  {name:35s} {r['n_params']:>9,} {r['final_val']:>9.4f} "
              f"{r['final_ppl']:>7.2f} {delta:>+14.1f}%{marker}")

    # ── Ablation analysis ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("ABLATION: What each fix contributes vs V6-baseline")
    print(SEP)
    base = results["V6-baseline"]["final_val"]
    fix_names = {
        "V6-F1 (content gate)": "Content-based gating (Fix 1)",
        "V6-F2 (contextual WG)": "Contextual write gate (Fix 2)",
        "V6-F3 (no pos_embed)": "Remove pos_embed (Fix 3)",
        "V6-F4 (SwiGLU FFN)":   "SwiGLU FFN (Fix 4)",
        "V6-F5 (anchor bias)":   "Anchor bias init (Fix 5)",
        "V6-F6 (sep. gates)":    "Separate gates (Fix 6)",
    }
    for name, desc in fix_names.items():
        if name not in results: continue
        delta = (results[name]['final_val'] - base) / base * 100
        sign  = "+" if delta >= 0 else ""
        helps = "HELPS" if delta < -0.5 else ("neutral" if abs(delta) < 0.5 else "HURTS")
        print(f"  {desc:38s}: {delta:+.2f}%  {helps}")

    # ── Best model generation ─────────────────────────────────────────────────
    best_name  = min(results, key=lambda k: results[k]['final_val'])
    best_model = dict(variants)[best_name]
    print(f"\n{SEP}")
    print(f"GENERATED TEXT - best model: {best_name}")
    print(SEP)
    print(generate(best_model, c2i, i2c))

    # ── Transformer generation for comparison ─────────────────────────────────
    tr_model = dict(variants)["Transformer"]
    print(f"\n── Transformer (for comparison) ──")
    print(generate(tr_model, c2i, i2c))

    # Save
    with open("v6_optim_results.json", 'w') as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv,(float,int))
                             else [float(x) for x in vv] if isinstance(vv,list)
                             else vv)
                        for kk,vv in v.items()}
                   for k,v in results.items()}, f, indent=2)
    print(f"\n  Saved -> v6_optim_results.json")
    print("  Done.")


if __name__ == '__main__':
    main()

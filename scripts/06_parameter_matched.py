"""
================================================================================
DUAL-GATE NEURON - EXPERIMENT 6: THE DEFINITIVE TEST
================================================================================

Two experiments that close the two biggest open questions:

EXPERIMENT A - Parameter-matched comparison (T=32)
  V6 has 43% more params than Transformer at n_emb=128.
  Critics can argue the advantage comes from capacity, not architecture.
  Solution: reduce V6's n_emb to 107 → ~807K params ≈ Transformer's 804K.
  If V6 still wins at matched params, the advantage is architectural.

EXPERIMENT B - Long context (T=128) with optimised V6
  At T=128, V6 trails Transformer by 3.8% (Table 6).
  We test with more anchors (N=16) and proper tau initialisation.
  If V6 wins at T=128, the context length limitation is resolved.

Both experiments: 3 seeds each, Welch's t-test, full statistical report.

Hardware: RTX 4080 (12GB VRAM), ~45 min total.
================================================================================
"""

import math, time, json, os, tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ──────────────────────────────────────────────────────────────────────────────

SEEDS = [42, 43, 44]

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    import requests
    url   = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache = os.path.join(tempfile.gettempdir(), "shakespeare.txt")
    if not os.path.exists(cache):
        print("Downloading TinyShakespeare...")
        resp = requests.get(url, timeout=30)
        with open(cache, 'w', encoding='utf-8') as f: f.write(resp.text)
    with open(cache, encoding='utf-8') as f: text = f.read()
    chars = sorted(set(text))
    c2i   = {c: i for i, c in enumerate(chars)}
    data  = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    print(f"Vocab: {len(chars)} | Train: {split:,} | Val: {len(data)-split:,}")
    return data[:split], data[split:], len(chars)


def get_batch(data, T, bs):
    ix = torch.randint(len(data) - T, (bs,))
    x  = torch.stack([data[i:i+T  ] for i in ix]).to(device)
    y  = torch.stack([data[i+1:i+T+1] for i in ix]).to(device)
    return x, y


# ──────────────────────────────────────────────────────────────────────────────
# CORE: causal EMA via conv1d
# ──────────────────────────────────────────────────────────────────────────────

def causal_ema(b: torch.Tensor, tau_raw: torch.Tensor) -> torch.Tensor:
    """EMA via causal conv1d: h_t = α·h_{t-1} + b_t, α = exp(-softplus(τ))."""
    B, T, C = b.shape
    tau  = F.softplus(tau_raw)
    j    = torch.arange(T, device=b.device, dtype=b.dtype)
    kern = torch.exp(-tau * j)
    b_t  = b.permute(0, 2, 1).contiguous()
    b_pad = F.pad(b_t, (T - 1, 0))
    k_w  = kern.flip(0).view(1, 1, T).expand(C, 1, -1)
    out  = F.conv1d(b_pad, k_w, groups=C)
    return out.permute(0, 2, 1)


def _tau_init(T: int) -> float:
    """tau_raw init so α^(T-1) >= 0.1."""
    alpha = 0.1 ** (1.0 / max(T - 1, 1))
    tau   = -math.log(max(alpha, 1e-10))
    return math.log(math.exp(tau) - 1 + 1e-10)


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
        self.register_buffer('mask', torch.tril(torch.ones(T, T)).view(1, 1, T, T))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        a = a.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        a = self.drop(F.softmax(a, dim=-1))
        return self.proj((a @ v).transpose(1, 2).contiguous().view(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, n_emb, n_heads, T, dropout):
        super().__init__()
        self.n1  = nn.LayerNorm(n_emb)
        self.att = CausalSelfAttention(n_emb, n_heads, T, dropout)
        self.n2  = nn.LayerNorm(n_emb)
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4*n_emb), nn.GELU(),
            nn.Linear(4*n_emb, n_emb), nn.Dropout(dropout))
        nn.init.normal_(self.att.proj.weight, std=0.02 / math.sqrt(2 * 4))
        nn.init.normal_(self.mlp[-2].weight,  std=0.02 / math.sqrt(2 * 4))

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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.emb(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# V6-TRUE-FAST (configurable n_emb for parameter matching)
# ──────────────────────────────────────────────────────────────────────────────

class V6Layer(nn.Module):
    """V6 layer with all 5 design optimisations (F1, F2, F3, F5, F6)."""

    def __init__(self, n_emb: int, T: int, T_f: int,
                 N_anchors: int, dropout: float):
        super().__init__()
        self.T  = T
        self.Tf = min(T_f, T)
        self.N  = N_anchors

        # Tau params
        self.tau_slow_raw = nn.Parameter(torch.tensor(_tau_init(T)))
        self.tau_fast_raw = nn.Parameter(torch.tensor(_tau_init(T_f)))
        self.tau_gate_raw = nn.Parameter(torch.tensor(_tau_init(max(T // 2, 4))))

        # Input projection
        self.w_proj = nn.Linear(n_emb, n_emb)

        # F6: separate write gates with asymmetric bias
        self.w_wx_f  = nn.Linear(n_emb, n_emb)
        nn.init.constant_(self.w_wx_f.bias, -0.5)
        self.w_wx_s  = nn.Linear(n_emb, n_emb)
        nn.init.constant_(self.w_wx_s.bias,  0.5)
        # F2: context signal from separate EMA
        self.w_ctx_f = nn.Linear(n_emb, n_emb, bias=False)
        self.w_ctx_s = nn.Linear(n_emb, n_emb, bias=False)

        # F1: content gate (lightweight)
        dk = max(n_emb // 8, 16)
        self.cg_q     = nn.Linear(n_emb, dk, bias=False)
        self.cg_k     = nn.Linear(n_emb, dk, bias=False)
        self.cg_scale = dk ** -0.5
        self.register_buffer('cg_mask',
            torch.tril(torch.ones(T, T)).view(1, T, T))

        # Output projections
        self.w_fast = nn.Linear(n_emb, n_emb)
        self.w_slow = nn.Linear(n_emb, n_emb)
        self.w_soma = nn.Linear(n_emb, n_emb)

        # F5: anchor shortcuts with biased init
        self.w_anchor      = nn.Linear(n_emb, n_emb)
        self.anchor_scales = nn.Parameter(torch.linspace(1.5, -1.5, N_anchors))

        self.blend_raw = nn.Parameter(torch.zeros(1))
        self.norm      = nn.LayerNorm(n_emb)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Input projection
        x_proj = torch.tanh(self.w_proj(x.reshape(B*T, C))).reshape(B, T, C)

        # F2: context signal (separate EMA, fully parallel)
        ctx_gate = causal_ema(x, self.tau_gate_raw)

        # F6: separate write gates
        def wgate(w_wx, w_ctx):
            return torch.sigmoid(
                w_wx(x.reshape(B*T, C)).reshape(B, T, C) +
                w_ctx(ctx_gate.reshape(B*T, C)).reshape(B, T, C))

        wg_f = wgate(self.w_wx_f, self.w_ctx_f)
        wg_s = wgate(self.w_wx_s, self.w_ctx_s)

        # Dual-timescale EMA (parallel via conv1d)
        ctx_fast = causal_ema(x_proj * wg_f, self.tau_fast_raw)
        ctx_slow = causal_ema(x_proj * wg_s, self.tau_slow_raw)

        # F1: content gate
        q = self.cg_q(x); k = self.cg_k(x)
        sc = (q @ k.transpose(-2, -1)) * self.cg_scale
        sc = sc.masked_fill(self.cg_mask[:, :T, :T] == 0, float('-inf'))
        cg = torch.sigmoid(sc.max(dim=-1).values.unsqueeze(-1))
        ctx_fast = ctx_fast * cg
        ctx_slow = ctx_slow * cg

        # F5: multi-anchor shortcuts
        scales = torch.sigmoid(self.anchor_scales)
        for i in range(self.N):
            pos = min(i * (T // self.N), T - 1)
            a = torch.tanh(self.w_anchor(x[:, pos, :]))
            ctx_fast = ctx_fast + scales[i] * a.unsqueeze(1)
            ctx_slow = ctx_slow + scales[i] * a.unsqueeze(1)

        # Gate blend + soma
        b    = torch.sigmoid(self.blend_raw)
        gate = torch.lerp(torch.sigmoid(self.w_fast(ctx_fast)),
                          torch.sigmoid(self.w_slow(ctx_slow)), b)
        soma = torch.tanh(self.w_soma(x.reshape(B*T, C))).reshape(B, T, C)
        return self.drop(self.norm(soma * gate))


class V6Block(nn.Module):
    def __init__(self, n_emb, T, T_f, N_anchors, dropout):
        super().__init__()
        self.n1  = nn.LayerNorm(n_emb)
        self.v6  = V6Layer(n_emb, T, T_f, N_anchors, dropout)
        self.n2  = nn.LayerNorm(n_emb)
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4*n_emb), nn.GELU(),
            nn.Linear(4*n_emb, n_emb), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.v6(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class V6LM(nn.Module):
    """V6 language model. No positional embeddings (F3)."""
    label = "V6"

    def __init__(self, vocab, n_emb, T, n_layers, n_heads, dropout,
                 T_f=8, N_anchors=4, **kw):
        super().__init__()
        self.T    = T
        self.emb  = nn.Embedding(vocab, n_emb)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            V6Block(n_emb, T, T_f, N_anchors, dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab, bias=False)
        self.emb.weight = self.head.weight
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.emb(idx))
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss(model, val_data, T, bs, n_batches=200):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_data, T, bs)
        logits = model(x)
        losses.append(F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)).item())
    model.train()
    return float(np.mean(losses))


def train(model, name, train_data, val_data, cfg, seed):
    torch.manual_seed(seed)
    # Re-init weights with this seed
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    model = model.to(device)
    T, bs = cfg['T'], cfg['batch_size']

    opt = torch.optim.AdamW(model.parameters(),
              lr=cfg['lr'], weight_decay=cfg['weight_decay'],
              betas=(0.9, 0.95))

    def lr_fn(s):
        if s < cfg['warmup']: return s / cfg['warmup']
        p = (s - cfg['warmup']) / max(1, cfg['n_steps'] - cfg['warmup'])
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    tok, t0 = 0, time.time()
    print(f"\n{'─'*60}")
    print(f"  {name}  seed={seed}  ({model.n_params():,} params)")
    print(f"{'─'*60}")

    model.train()
    for step in range(cfg['n_steps']):
        x, y = get_batch(train_data, T, bs)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        opt.step(); sched.step()
        tok += bs * T

        if (step + 1) % cfg['eval_every'] == 0:
            vl = eval_loss(model, val_data, T, bs)
            el = time.time() - t0
            print(f"  step {step+1:5d} | val={vl:.4f} ppl={math.exp(vl):.2f}"
                  f" | tok/s={tok/el:,.0f}")

    final = eval_loss(model, val_data, T, bs, n_batches=800)
    elapsed = time.time() - t0
    tps = tok / elapsed
    print(f"\n  FINAL val={final:.4f} ppl={math.exp(final):.2f}"
          f" | {elapsed:.0f}s | {tps:,.0f} tok/s")

    # Extract learned alphas
    alphas = {}
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'v6'):
            af = torch.exp(-F.softplus(block.v6.tau_fast_raw)).item()
            asl = torch.exp(-F.softplus(block.v6.tau_slow_raw)).item()
            alphas[f'layer_{i}'] = {
                'alpha_fast': round(af, 4),
                'alpha_slow': round(asl, 4),
                'eff_mem_fast': round(1/(1-af+1e-8), 1),
                'eff_mem_slow': round(1/(1-asl+1e-8), 1),
            }

    return dict(
        final_val=final, final_ppl=math.exp(final),
        time_sec=elapsed, tok_per_sec=tps,
        n_params=model.n_params(), seed=seed,
        alphas=alphas,
    )


# ──────────────────────────────────────────────────────────────────────────────
# STATISTICAL ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def welch_t_test(a, b):
    """Two-sample Welch's t-test (unequal variances)."""
    na, nb = len(a), len(b)
    ma, mb = np.mean(a), np.mean(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(va/na + vb/nb)
    t_stat = (ma - mb) / se
    df_num = (va/na + vb/nb)**2
    df_den = (va/na)**2/(na-1) + (vb/nb)**2/(nb-1)
    df = df_num / df_den
    # Two-tailed p-value approximation
    from math import gamma, pi
    def t_cdf(t, v):
        x = v / (v + t*t)
        # Regularised incomplete beta via simple approximation
        # For large |t|, p ≈ 0; good enough for our purposes
        if abs(t) > 10: return 0.0 if t > 0 else 1.0
        # Numerical integration (simple)
        n_pts = 10000
        dt = abs(t) / n_pts
        s = 0.0
        c = gamma((v+1)/2) / (gamma(v/2) * math.sqrt(v * pi))
        for i in range(n_pts):
            ti = -abs(t) + (i + 0.5) * (2*abs(t)) / n_pts
            s += c * (1 + ti*ti/v)**(-(v+1)/2)
        p_less = s * (2*abs(t)) / n_pts
        return p_less
    p_two = 2 * min(t_cdf(t_stat, df), 1 - t_cdf(t_stat, df))
    # Use simpler approach: scipy-free p-value via normal approx for df > 30
    if df > 30:
        from math import erfc
        p_two = erfc(abs(t_stat) / math.sqrt(2))
    return t_stat, df, min(p_two, 1.0)


def bootstrap_ci(a, b, n_boot=10000, ci=0.95):
    """Bootstrap CI for relative difference (mean(a) - mean(b)) / mean(b)."""
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append((np.mean(sa) - np.mean(sb)) / np.mean(sb))
    lo = np.percentile(diffs, (1 - ci) / 2 * 100)
    hi = np.percentile(diffs, (1 + ci) / 2 * 100)
    return lo * 100, hi * 100  # as percentages


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return abs(np.mean(a) - np.mean(b)) / max(pooled_std, 1e-10)


def print_stats(name_a, vals_a, name_b, vals_b):
    """Print full statistical comparison."""
    ma, mb = np.mean(vals_a), np.mean(vals_b)
    delta_pct = (ma - mb) / mb * 100

    print(f"\n  {name_a}: {ma:.4f} ± {np.std(vals_a):.4f}  (PPL {math.exp(ma):.2f})")
    print(f"  {name_b}: {mb:.4f} ± {np.std(vals_b):.4f}  (PPL {math.exp(mb):.2f})")
    print(f"  Δ = {delta_pct:+.2f}%")

    if len(vals_a) >= 2 and len(vals_b) >= 2:
        t, df, p = welch_t_test(vals_a, vals_b)
        d = cohens_d(vals_a, vals_b)
        lo, hi = bootstrap_ci(vals_a, vals_b)
        print(f"  Welch's t = {t:.2f}, df = {df:.1f}, p = {p:.2e}")
        print(f"  Cohen's d = {d:.1f}")
        print(f"  Bootstrap 95% CI for Δ: [{lo:.1f}%, {hi:.1f}%]")

    # Check overlap
    worst_a = max(vals_a) if delta_pct < 0 else min(vals_a)
    best_b  = min(vals_b) if delta_pct < 0 else max(vals_b)
    if delta_pct < 0:
        overlap = worst_a > best_b
    else:
        overlap = worst_a < best_b
    print(f"  Distributions overlap: {'YES' if overlap else 'NO'}")
    print(f"  Per-seed: {[f'{v:.4f}' for v in vals_a]} vs {[f'{v:.4f}' for v in vals_b]}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(exp_name, cfg, transformer_cfg, v6_cfg):
    """Run one experiment with given configs."""
    SEP = "=" * 62
    print(f"\n{SEP}")
    print(f"  {exp_name}")
    print(f"  T={cfg['T']}, steps={cfg['n_steps']}, seeds={SEEDS}")
    print(f"{SEP}")

    train_data, val_data, vocab = load_data()

    # Build models to check param counts
    tr_model = TransformerLM(vocab=vocab, **transformer_cfg)
    v6_model = V6LM(vocab=vocab, **v6_cfg)
    tr_p = tr_model.n_params()
    v6_p = v6_model.n_params()
    print(f"\n  Transformer: {tr_p:,} params")
    print(f"  V6:          {v6_p:,} params")
    print(f"  Ratio:       {v6_p/tr_p:.3f}x")

    results = {'Transformer': [], 'V6': []}

    for seed in SEEDS:
        print(f"\n{'─'*62}\nSEED {seed}\n{'─'*62}")

        # Fresh models each seed
        tr_m = TransformerLM(vocab=vocab, **transformer_cfg)
        v6_m = V6LM(vocab=vocab, **v6_cfg)

        r_tr = train(tr_m, "Transformer", train_data, val_data, cfg, seed)
        results['Transformer'].append(r_tr)

        r_v6 = train(v6_m, "V6", train_data, val_data, cfg, seed)
        results['V6'].append(r_v6)

    # Statistical summary
    tr_vals = [r['final_val'] for r in results['Transformer']]
    v6_vals = [r['final_val'] for r in results['V6']]

    print(f"\n{SEP}")
    print(f"  RESULTS: {exp_name}")
    print(f"{SEP}")
    print_stats("V6", v6_vals, "Transformer", tr_vals)

    # Print alphas from last seed
    last_v6 = results['V6'][-1]
    if last_v6.get('alphas'):
        print(f"\n  Learned α values (seed {last_v6['seed']}):")
        for layer, vals in last_v6['alphas'].items():
            print(f"    {layer}: α_fast={vals['alpha_fast']:.3f} "
                  f"({vals['eff_mem_fast']:.1f} steps), "
                  f"α_slow={vals['alpha_slow']:.3f} "
                  f"({vals['eff_mem_slow']:.1f} steps)")
        # Verify biological prediction
        all_ok = all(v['alpha_slow'] > v['alpha_fast']
                     for v in last_v6['alphas'].values())
        print(f"    α_slow > α_fast in all layers: {'✓ YES' if all_ok else '✗ NO'}")

    # Throughput
    tr_tps = results['Transformer'][0]['tok_per_sec']
    v6_tps = results['V6'][0]['tok_per_sec']
    print(f"\n  Throughput: Transformer {tr_tps:,.0f} tok/s, V6 {v6_tps:,.0f} tok/s "
          f"({v6_tps/tr_tps:.2f}x)")

    return results


def main():
    MASTER_SEP = "█" * 62

    # ══════════════════════════════════════════════════════════════
    # EXPERIMENT A: Parameter-matched at T=32
    # ══════════════════════════════════════════════════════════════
    print(f"\n{MASTER_SEP}")
    print("  EXPERIMENT A: PARAMETER-MATCHED COMPARISON (T=32)")
    print(f"  V6 n_emb=107 (~807K) vs Transformer n_emb=128 (~804K)")
    print(f"{MASTER_SEP}")

    cfg_a = dict(
        T=32, batch_size=128, n_steps=5000, eval_every=500,
        lr=3e-4, weight_decay=0.1, grad_clip=1.0, warmup=200,
    )
    tr_cfg_a = dict(
        n_emb=128, T=32, n_layers=4, n_heads=4, dropout=0.1,
    )
    v6_cfg_a = dict(
        n_emb=107, T=32, n_layers=4, n_heads=4, dropout=0.1,
        T_f=8, N_anchors=4,
    )

    results_a = run_experiment(
        "EXPERIMENT A: Parameter-matched (T=32)",
        cfg_a, tr_cfg_a, v6_cfg_a)

    # ══════════════════════════════════════════════════════════════
    # EXPERIMENT B: Long context T=128
    # ══════════════════════════════════════════════════════════════
    print(f"\n{MASTER_SEP}")
    print("  EXPERIMENT B: LONG CONTEXT (T=128)")
    print(f"  V6 n_emb=128, N_anchors=16 vs Transformer n_emb=128")
    print(f"{MASTER_SEP}")

    cfg_b = dict(
        T=128, batch_size=64, n_steps=5000, eval_every=500,
        lr=3e-4, weight_decay=0.1, grad_clip=1.0, warmup=200,
    )
    tr_cfg_b = dict(
        n_emb=128, T=128, n_layers=4, n_heads=4, dropout=0.1,
    )
    v6_cfg_b = dict(
        n_emb=128, T=128, n_layers=4, n_heads=4, dropout=0.1,
        T_f=16, N_anchors=16,
    )

    results_b = run_experiment(
        "EXPERIMENT B: Long context (T=128)",
        cfg_b, tr_cfg_b, v6_cfg_b)

    # ══════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════════
    print(f"\n{MASTER_SEP}")
    print("  FINAL VERDICT")
    print(f"{MASTER_SEP}")

    for exp_name, results in [("A (param-matched T=32)", results_a),
                                ("B (T=128)", results_b)]:
        tr_mean = np.mean([r['final_val'] for r in results['Transformer']])
        v6_mean = np.mean([r['final_val'] for r in results['V6']])
        delta = (v6_mean - tr_mean) / tr_mean * 100
        winner = "V6" if v6_mean < tr_mean else "Transformer"
        tr_ppl = math.exp(tr_mean)
        v6_ppl = math.exp(v6_mean)
        print(f"\n  Experiment {exp_name}:")
        print(f"    Transformer PPL: {tr_ppl:.2f}  |  V6 PPL: {v6_ppl:.2f}")
        print(f"    Δ = {delta:+.2f}%  →  Winner: {winner}")

    # Save results
    out = {}
    for exp_name, results in [("param_matched_T32", results_a),
                                ("long_context_T128", results_b)]:
        out[exp_name] = {}
        for model_name, runs in results.items():
            vals = [r['final_val'] for r in runs]
            out[exp_name][model_name] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'ppl': float(math.exp(np.mean(vals))),
                'params': runs[0]['n_params'],
                'seeds': [float(v) for v in vals],
                'tok_per_sec': float(runs[0]['tok_per_sec']),
            }

    fname = os.path.join(tempfile.gettempdir(), "v6_experiment6_results.json")
    with open(fname, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved → {fname}")
    print(f"\n{'█'*62}")
    print("  DONE")
    print(f"{'█'*62}")


if __name__ == '__main__':
    main()

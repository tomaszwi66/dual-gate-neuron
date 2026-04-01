"""
================================================================================
DUAL-GATE NEURON - Neuron Evolution Benchmark
V1 → V5: each version adds one biological mechanism
================================================================================

Architecture evolution:

  V1  Perceptron          - baseline, no temporal memory
  V2  Decay gate          - exponential decay weighted context
  V3  Dual EMA            - two timescales (fast α≈0.27, slow α≈0.88)
  V4  Projected dual EMA  - input projection before EMA (like LSTM)
  V5  STDP + dual EMA     - temporal correlation (Hebbian / STDP)

Each version tested on three tasks:
  XOR    - Delayed Sign-XOR (Level 1: architectural necessity)
  LAG    - Multi-lag regression (Level 2: efficiency)
  MNIST  - Sequential MNIST (Level 3: generality)

Biological inspirations:
  Dual timescale → Losonczy & Magee 2006 (fast/slow dendritic spikes)
  Projection     → Larkum 2013 (compartment-specific processing)
  STDP           → Bi & Poo 1998 (timing-dependent synaptic plasticity)
  EMA trace      → Francioni et al. 2026 (apical calcium integration)
================================================================================
"""

import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

SEED   = 42
T      = 8       # sequence length for XOR + LAG tasks
N_IN   = 6       # input features
N_HID  = 32      # hidden dimension (same for all models)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────

def make_xor(n: int, seed: int = 0):
    """Delayed Sign-XOR: target = same sign at step 0 and step T-1."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, T, N_IN, generator=g)
    y = ((x[:, 0, 0] * x[:, -1, 0]) > 0).float()
    return x.to(device), y.to(device)


def make_lag(n: int, seed: int = 0):
    """Multi-lag regression: target depends on lags 1, 3, 5."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, T, N_IN, generator=g)
    y = (
        x[:, -1, 0]
        + x[:, -3, 1] * 0.7
        + x[:, -5, 2] * 0.5
        + torch.randn(n, generator=g) * 0.1
    ).unsqueeze(1)
    return x.to(device), y.to(device)


def load_mnist():
    """Sequential MNIST: 784 pixels as a time series, 10 classes."""
    try:
        from torchvision import datasets, transforms
        tr = datasets.MNIST(os.path.join(tempfile.gettempdir(), "mnist"), train=True,  download=True,
                            transform=transforms.ToTensor())
        te = datasets.MNIST(os.path.join(tempfile.gettempdir(), "mnist"), train=False, download=True,
                            transform=transforms.ToTensor())
        N_TR, N_TE = 10000, 2000
        x_tr = tr.data[:N_TR].float().view(N_TR, 784, 1) / 255.0
        y_tr = tr.targets[:N_TR].long()
        x_te = te.data[:N_TE].float().view(N_TE, 784, 1) / 255.0
        y_te = te.targets[:N_TE].long()
        return x_tr.to(device), y_tr.to(device), x_te.to(device), y_te.to(device), True
    except Exception as e:
        print(f"  MNIST unavailable ({e}), skipping Level 3.")
        return None, None, None, None, False


# Pre-generate data once
XOR_TR, Y_XOR_TR = make_xor(4000, seed=0)
XOR_TE, Y_XOR_TE = make_xor(1000, seed=99)
LAG_TR, Y_LAG_TR = make_lag(5000, seed=0)
LAG_TE, Y_LAG_TE = make_lag(1000, seed=99)


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_classifier(
    model: nn.Module,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    epochs: int = 80,
    batch:  int = 128,
    lr:     float = 3e-3,
    conv_thr: float = 0.80,
) -> dict:
    """Binary classifier training. Returns test accuracy + convergence epoch."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ld  = DataLoader(TensorDataset(x_tr, y_tr), batch, shuffle=True,
                     generator=torch.Generator().manual_seed(SEED))
    best_acc, conv_ep = 0.0, epochs

    for ep in range(epochs):
        model.train()
        for bx, by in ld:
            loss = F.binary_cross_entropy_with_logits(
                model(bx).squeeze(1), by)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            acc = (model(x_te).squeeze(1) > 0).eq(y_te > 0.5).float().mean().item()
        best_acc = max(best_acc, acc)
        if acc >= conv_thr and conv_ep == epochs:
            conv_ep = ep + 1

    return dict(metric=best_acc, conv=conv_ep, params=count_params(model))


def train_regressor(
    model: nn.Module,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    epochs:   int   = 80,
    batch:    int   = 256,
    lr:       float = 3e-3,
    conv_thr: float = 0.30,
) -> dict:
    """Regression training. Returns test MSE + convergence epoch."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ld  = DataLoader(TensorDataset(x_tr, y_tr), batch, shuffle=True,
                     generator=torch.Generator().manual_seed(SEED))
    best_mse, conv_ep = 1e9, epochs

    for ep in range(epochs):
        model.train()
        for bx, by in ld:
            loss = F.mse_loss(model(bx), by)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            mse = F.mse_loss(model(x_te), y_te).item()
        best_mse = min(best_mse, mse)
        if mse <= conv_thr and conv_ep == epochs:
            conv_ep = ep + 1

    return dict(metric=best_mse, conv=conv_ep, params=count_params(model))


def train_multiclass(
    model: nn.Module,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    epochs: int   = 15,
    batch:  int   = 256,
    lr:     float = 3e-3,
) -> dict:
    """Multi-class classification training. Returns test accuracy."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ld  = DataLoader(TensorDataset(x_tr, y_tr), batch, shuffle=True,
                     generator=torch.Generator().manual_seed(SEED))
    best_acc = 0.0

    for ep in range(epochs):
        model.train()
        for bx, by in ld:
            loss = F.cross_entropy(model(bx), by)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            acc = model(x_te).argmax(1).eq(y_te).float().mean().item()
        best_acc = max(best_acc, acc)

    return dict(metric=best_acc, conv=epochs, params=count_params(model))


# ──────────────────────────────────────────────────────────────────────────────
# NEURON ARCHITECTURES  V1 → V5
# All models share the same interface:
#   __init__(n_in, n_hid, T, n_out)
#   forward(x: Tensor[B, T, n_in]) → Tensor[B, n_out]
# ──────────────────────────────────────────────────────────────────────────────

class V1_Perceptron(nn.Module):
    """Baseline: sees only x[:, -1, :]. No temporal memory at all.

    Biological analogy: a single integrate-and-fire neuron with no
    dendritic computation and no synaptic history.
    """
    label = "V1  Perceptron (baseline)"

    def __init__(self, n_in, n_hid, T, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hid), nn.ReLU(),
            nn.Linear(n_hid, n_out),
        )

    def forward(self, x):
        return self.net(x[:, -1, :])


class V2_DecayGate(nn.Module):
    """Exponential decay weighted context - single timescale.

    Biological analogy: passive membrane decay with fixed time constant.
    The neuron sees a blur of the past, but cannot distinguish what was
    important.

    Key limitation: tau is learnable but all timesteps share the same
    exponential weighting - no content-based selection.
    """
    label = "V2  Decay gate (single τ)"

    def __init__(self, n_in, n_hid, T, n_out):
        super().__init__()
        self.T       = T
        self.tau_raw = nn.Parameter(torch.tensor(0.17))
        self.w_ctx   = nn.Linear(n_in, n_hid)
        self.w_soma  = nn.Linear(n_in, n_hid)
        self.head    = nn.Linear(n_hid, n_out)
        self.norm    = nn.LayerNorm(n_hid)

    def forward(self, x):
        B, T, C = x.shape
        tau = F.softplus(self.tau_raw)                          # tau > 0, gradient flows through softplus
        pos = torch.arange(T, device=x.device, dtype=x.dtype)
        # Softmax over decay ensures weights sum to 1
        w   = F.softmax(-tau * (T - 1 - pos), dim=0)           # (T,)
        ctx = (x * w.view(1, T, 1)).sum(1)                      # (B, C)
        soma = torch.tanh(self.w_soma(x[:, -1, :]))
        gate = torch.sigmoid(self.w_ctx(ctx))
        return self.head(self.norm(soma * gate))


class V3_DualEMA(nn.Module):
    """Two independent EMA tracks: fast (short memory) + slow (long memory).

    Biological analogy: fast Na+ dendritic spikes (proximal dendrites,
    short time constant ~5ms) vs slow Ca²+ plateau potentials (apical
    dendrites, long time constant ~100ms).
    Reference: Losonczy & Magee 2006.

    Key improvement over V2: the two timescales are INDEPENDENT and
    learnable, each tracking a different aspect of history.
    """
    label = "V3  Dual EMA (fast+slow)"

    def __init__(self, n_in, n_hid, T, n_out):
        super().__init__()
        self.T      = T
        # sigmoid(-1) ≈ 0.27 (fast decay), sigmoid(2) ≈ 0.88 (slow decay)
        self.alpha_fast = nn.Parameter(torch.tensor(-1.0))
        self.alpha_slow = nn.Parameter(torch.tensor( 2.0))
        self.w_fast  = nn.Linear(n_in, n_hid)
        self.w_slow  = nn.Linear(n_in, n_hid)
        self.w_soma  = nn.Linear(n_in, n_hid)
        self.blend   = nn.Parameter(torch.zeros(1))
        self.head    = nn.Linear(n_hid, n_out)
        self.norm    = nn.LayerNorm(n_hid)

    def _ema(self, x: torch.Tensor, alpha_raw: torch.Tensor) -> torch.Tensor:
        """Exponential moving average over the time dimension.

        h_t = α · h_{t-1} + (1-α) · x_t
        Returns h_T - the trace after seeing all T timesteps.
        Gradient flows through α via sigmoid(alpha_raw).
        """
        alpha = torch.sigmoid(alpha_raw)
        h = torch.zeros(x.shape[0], x.shape[2], device=x.device)
        for t in range(x.shape[1]):
            h = alpha * h + (1.0 - alpha) * x[:, t, :]
        return h

    def forward(self, x):
        h_fast = self._ema(x, self.alpha_fast)
        h_slow = self._ema(x, self.alpha_slow)
        g_fast = torch.sigmoid(self.w_fast(h_fast))
        g_slow = torch.sigmoid(self.w_slow(h_slow))
        blend  = torch.sigmoid(self.blend)
        gate   = torch.lerp(g_fast, g_slow, blend)
        soma   = torch.tanh(self.w_soma(x[:, -1, :]))
        return self.head(self.norm(soma * gate))


class V4_ProjectedDualEMA(nn.Module):
    """Projected dual EMA: linear projection of input before EMA.

    Biological analogy: synaptic weights transform the pre-synaptic signal
    before it enters the dendritic compartment. The dendrite does not
    operate on raw spike rates but on synaptic currents.
    Reference: Larkum 2013 (compartment-specific processing).

    Key improvement over V3: the EMA now operates in a *learned feature
    space* (n_hid dimensions), not the raw input space (n_in dimensions).
    This allows the model to select which features to track over time.
    """
    label = "V4  Projected dual EMA"

    def __init__(self, n_in, n_hid, T, n_out):
        super().__init__()
        self.T          = T
        self.alpha_fast = nn.Parameter(torch.tensor(-1.0))
        self.alpha_slow = nn.Parameter(torch.tensor( 2.0))
        self.proj       = nn.Linear(n_in, n_hid)     # input projection
        self.w_fast     = nn.Linear(n_hid, n_hid)
        self.w_slow     = nn.Linear(n_hid, n_hid)
        self.w_soma     = nn.Linear(n_in,  n_hid)
        self.blend      = nn.Parameter(torch.zeros(1))
        self.head       = nn.Linear(n_hid, n_out)
        self.norm       = nn.LayerNorm(n_hid)

    def _projected_ema(self, x_proj: torch.Tensor,
                       alpha_raw: torch.Tensor) -> torch.Tensor:
        """EMA in projected feature space."""
        alpha = torch.sigmoid(alpha_raw)
        h = torch.zeros(x_proj.shape[0], x_proj.shape[2], device=x_proj.device)
        for t in range(x_proj.shape[1]):
            h = alpha * h + (1.0 - alpha) * x_proj[:, t, :]
        return h

    def forward(self, x):
        B, T, C = x.shape
        # Project all timesteps: (B*T, C) → (B*T, n_hid) → (B, T, n_hid)
        x_proj = torch.tanh(
            self.proj(x.reshape(B * T, C))
        ).reshape(B, T, -1)

        h_fast = self._projected_ema(x_proj, self.alpha_fast)
        h_slow = self._projected_ema(x_proj, self.alpha_slow)
        g_fast = torch.sigmoid(self.w_fast(h_fast))
        g_slow = torch.sigmoid(self.w_slow(h_slow))
        blend  = torch.sigmoid(self.blend)
        gate   = torch.lerp(g_fast, g_slow, blend)
        soma   = torch.tanh(self.w_soma(x[:, -1, :]))
        return self.head(self.norm(soma * gate))


class V5_STDPDualEMA(nn.Module):
    """STDP-inspired correlation + projected dual EMA.

    Biological analogy: Spike-Timing-Dependent Plasticity (STDP) - synaptic
    strength changes proportionally to the temporal correlation between
    pre- and post-synaptic firing. A synapse is strengthened when the
    pre-synaptic neuron fires just before the post-synaptic one.
    Reference: Bi & Poo 1998.

    Implementation: the STDP term is the element-wise product of:
        - x_proj[:, -1, :]  - current projected activity (post-synaptic)
        - h_slow             - slow EMA trace (pre-synaptic history)
    This gives the neuron a signal about WHICH features co-activated
    across time - directly encoding temporal correlations.

    Key improvement over V4: the network can now detect *which* past
    events are correlated with the present, not just how much history
    there was.
    """
    label = "V5  STDP + projected dual EMA"

    def __init__(self, n_in, n_hid, T, n_out):
        super().__init__()
        self.T          = T
        self.alpha_fast = nn.Parameter(torch.tensor(-1.0))
        self.alpha_slow = nn.Parameter(torch.tensor( 2.0))
        self.proj       = nn.Linear(n_in,  n_hid)
        self.w_fast     = nn.Linear(n_hid, n_hid)
        self.w_slow     = nn.Linear(n_hid, n_hid)
        self.w_corr     = nn.Linear(n_hid, n_hid)  # STDP correlation path
        self.w_soma     = nn.Linear(n_in,  n_hid)
        self.blend_hs   = nn.Parameter(torch.zeros(1))  # fast vs slow
        self.blend_corr = nn.Parameter(torch.zeros(1))  # history vs correlation
        self.head       = nn.Linear(n_hid, n_out)
        self.norm       = nn.LayerNorm(n_hid)

    def _projected_ema(self, x_proj: torch.Tensor,
                       alpha_raw: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(alpha_raw)
        h = torch.zeros(x_proj.shape[0], x_proj.shape[2], device=x_proj.device)
        for t in range(x_proj.shape[1]):
            h = alpha * h + (1.0 - alpha) * x_proj[:, t, :]
        return h

    def forward(self, x):
        B, T, C = x.shape
        x_proj  = torch.tanh(
            self.proj(x.reshape(B * T, C))
        ).reshape(B, T, -1)                                    # (B, T, n_hid)

        h_fast  = self._projected_ema(x_proj, self.alpha_fast)
        h_slow  = self._projected_ema(x_proj, self.alpha_slow)

        # STDP correlation: current projected activity × slow history (Bi & Poo 1998)
        x_now   = x_proj[:, -1, :]                            # (B, n_hid)
        corr    = x_now * h_slow                               # element-wise

        # Gate from history track
        b_hs   = torch.sigmoid(self.blend_hs)
        g_hist = torch.lerp(
            torch.sigmoid(self.w_fast(h_fast)),
            torch.sigmoid(self.w_slow(h_slow)),
            b_hs,
        )

        # Gate from correlation track
        g_corr = torch.sigmoid(self.w_corr(corr))

        # Final blend: history vs STDP correlation
        b_c    = torch.sigmoid(self.blend_corr)
        gate   = torch.lerp(g_hist, g_corr, b_c)

        soma   = torch.tanh(self.w_soma(x[:, -1, :]))
        return self.head(self.norm(soma * gate))


# LSTM reference (full backprop, cuDNN)
class LSTMReference(nn.Module):
    """Industry-standard LSTM baseline with full BPTT."""
    label = "LSTM (reference)"

    def __init__(self, n_in, n_hid, T, n_out):
        super().__init__()
        self.lstm = nn.LSTM(n_in, n_hid, batch_first=True)
        self.head = nn.Linear(n_hid, n_out)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


# ──────────────────────────────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ──────────────────────────────────────────────────────────────────────────────

ARCHITECTURES = [
    V1_Perceptron,
    V2_DecayGate,
    V3_DualEMA,
    V4_ProjectedDualEMA,
    V5_STDPDualEMA,
    LSTMReference,
]


def run_xor() -> dict:
    """Level 1: Delayed Sign-XOR."""
    print(f"\n{'='*64}")
    print("LEVEL 1 - DELAYED SIGN-XOR  (necessity proof)")
    print(f"{'='*64}")
    results = {}
    for Cls in ARCHITECTURES:
        m = Cls(N_IN, N_HID, T, 1)
        r = train_classifier(m, XOR_TR, Y_XOR_TR, XOR_TE, Y_XOR_TE,
                             epochs=80, batch=128, lr=3e-3, conv_thr=0.80)
        results[Cls.label] = r
        conv = f"ep {r['conv']}" if r['conv'] < 80 else "never"
        mark = " ***" if r['metric'] > 0.80 else ""
        print(f"  {Cls.label:35s}  acc={r['metric']:.4f}  conv@{conv}  "
              f"params={r['params']:,}{mark}")
    return results


def run_lag() -> dict:
    """Level 2: Multi-lag regression."""
    print(f"\n{'='*64}")
    print("LEVEL 2 - MULTI-LAG REGRESSION  (efficiency proof)")
    print(f"{'='*64}")
    results = {}
    for Cls in ARCHITECTURES:
        m = Cls(N_IN, N_HID, T, 1)
        r = train_regressor(m, LAG_TR, Y_LAG_TR, LAG_TE, Y_LAG_TE,
                            epochs=80, batch=256, lr=3e-3, conv_thr=0.30)
        results[Cls.label] = r
        conv = f"ep {r['conv']}" if r['conv'] < 80 else "never"
        mark = " ***" if r['metric'] < 0.15 else ""
        print(f"  {Cls.label:35s}  mse={r['metric']:.4f}  conv@{conv}  "
              f"params={r['params']:,}{mark}")
    return results


def run_mnist(mnist_data: tuple) -> dict:
    """Level 3: Sequential MNIST."""
    x_tr, y_tr, x_te, y_te, ok = mnist_data
    if not ok:
        return {}

    T_mnist  = 784
    n_in_m   = 1
    n_hid_m  = 64

    print(f"\n{'='*64}")
    print("LEVEL 3 - SEQUENTIAL MNIST  (generality proof)")
    print(f"{'='*64}")
    results = {}
    for Cls in ARCHITECTURES:
        if Cls is V1_Perceptron:
            continue  # perceptron is meaningless on MNIST
        m = Cls(n_in_m, n_hid_m, T_mnist, 10)
        r = train_multiclass(m, x_tr, y_tr, x_te, y_te,
                             epochs=15, batch=256, lr=3e-3)
        results[Cls.label] = r
        mark = " ***" if r['metric'] > 0.85 else ""
        print(f"  {Cls.label:35s}  acc={r['metric']:.4f}  "
              f"params={r['params']:,}{mark}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY + RANKING
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(r_xor: dict, r_lag: dict, r_mnist: dict):
    """Print final ranking across all tasks."""
    SEP = "=" * 64
    print(f"\n{SEP}")
    print("FINAL SUMMARY - EVOLUTION RANKING")
    print(SEP)

    print(f"\n  {'Architecture':35s}  {'XOR acc':>8s}  {'Lag MSE':>8s}  {'MNIST':>6s}")
    print(f"  {'─'*63}")

    for Cls in ARCHITECTURES:
        label = Cls.label
        xor  = f"{r_xor[label]['metric']:.4f}" if label in r_xor else "  -  "
        lag  = f"{r_lag[label]['metric']:.4f}" if label in r_lag else "  -  "
        mn   = f"{r_mnist[label]['metric']:.4f}" if label in r_mnist else "  -  "
        print(f"  {label:35s}  {xor:>8s}  {lag:>8s}  {mn:>6s}")

    print(f"\n  Biological mechanisms added per version:")
    print(f"  V1: none (baseline)")
    print(f"  V2: + exponential decay weighted context  [Passive membrane RC]")
    print(f"  V3: + dual timescale EMA (fast + slow)    [Losonczy & Magee 2006]")
    print(f"  V4: + input projection before EMA          [Larkum 2013]")
    print(f"  V5: + STDP correlation gate                [Bi & Poo 1998]")

    print(f"\n  If V5 dominates: all three biological principles contribute.")
    print(f"  If V4=V5: STDP adds no benefit on these tasks.")
    print(f"  If V3=V4: projection is not necessary at this scale.")

    # V5 parameters after training (from lag task - most informative)
    print(f"\n  V5 learned biological parameters (from lag task):")
    v5_key = V5_STDPDualEMA.label
    if v5_key in r_lag:
        # Re-train V5 briefly to get final params
        m5 = V5_STDPDualEMA(N_IN, N_HID, T, 1).to(device)
        opt = torch.optim.AdamW(m5.parameters(), lr=3e-3)
        ld  = DataLoader(TensorDataset(LAG_TR, Y_LAG_TR), 256, shuffle=True,
                         generator=torch.Generator().manual_seed(SEED))
        for _ in range(20):
            for bx, by in ld:
                F.mse_loss(m5(bx), by).backward()
                opt.step(); opt.zero_grad()
        af = torch.sigmoid(m5.alpha_fast).item()
        as_ = torch.sigmoid(m5.alpha_slow).item()
        bhs = torch.sigmoid(m5.blend_hs).item()
        bc  = torch.sigmoid(m5.blend_corr).item()
        print(f"    alpha_fast = {af:.4f}  (effective memory: ~{1/(1-af+1e-8):.1f} steps)")
        print(f"    alpha_slow = {as_:.4f}  (effective memory: ~{1/(1-as_+1e-8):.1f} steps)")
        print(f"    blend_hs   = {bhs:.4f}  (0=fast, 1=slow)")
        print(f"    blend_corr = {bc:.4f}  (0=history, 1=STDP correlation)")
        if as_ > af:
            print(f"    => alpha_slow > alpha_fast: CONFIRMED (biological prediction)")
        else:
            print(f"    => WARNING: alpha ordering inverted")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("DUAL-GATE NEURON - NEURON EVOLUTION V1→V5
  Ablation study: each biological mechanism tested individually.")
    print("Each version adds one biological mechanism")
    print("=" * 64)

    mnist_data = load_mnist()

    r_xor   = run_xor()
    r_lag   = run_lag()
    r_mnist = run_mnist(mnist_data)

    print_summary(r_xor, r_lag, r_mnist)

    # Save
    out = {}
    for task, res in [("xor", r_xor), ("lag", r_lag), ("mnist", r_mnist)]:
        out[task] = {
            k: {kk: float(vv) if isinstance(vv, float) else int(vv)
                for kk, vv in v.items()}
            for k, v in res.items()
        }
    with open("evolution_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Saved → evolution_results.json")
    print("  Done.")


if __name__ == "__main__":
    main()

"""
Phase-SNN — PyTorch Implementation (Phase 3, clean)
=====================================================
Key change from previous versions:
  Complex weights replaced with two real matrices (W_real, W_imag).
  Mathematically identical, but:
  - No PyTorch complex dtype issues
  - Standard float16 mixed precision works
  - Standard Adam works (no conjugate tricks)
  - Faster GPU matmul (real tensors are better optimised)

Architecture:
  PhaseEncoderHead: W_real(K,D) + W_imag(K,D) + omega(K)
  MultiHeadPhaseEncoder: N_heads × PhaseEncoderHead → concat → proj
  PhaseNorm: layer norm in cos/sin space (avoids ±π discontinuity)
  PhaseFFN: cos/sin → Linear → GELU → Linear → residual (Phase 3 addition)
  PhaseScanLayer: Hillis-Steele scan + PhaseFFN + residuals
  PhaseLM: embedding → encoder → N×scan → norm → LM head
  PhaseIntentClassifier: encoder → mean-pool → hidden → softmax
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Learning rate schedule ────────────────────────────────────────────────────

def cosine_lr_schedule(optimizer, step, total_steps, lr_max,
                       lr_min=1e-5, warmup_steps=0):
    """Cosine annealing with linear warmup."""
    if warmup_steps > 0 and step < warmup_steps:
        lr = lr_max * step / max(warmup_steps, 1)
    else:
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr   = lr_min + 0.5*(lr_max - lr_min)*(1 + math.cos(math.pi*prog))
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


# ── Hillis-Steele parallel prefix scan ───────────────────────────────────────

def hillis_steele_scan(x: torch.Tensor) -> torch.Tensor:
    """
    Causal parallel prefix sum. O(log L) depth.
    x: (B, L, K) → output[t] = sum(x[0..t])
    Gives each position access to all past context.
    """
    B, L, K = x.shape
    res = x.clone()
    steps = max(1, math.ceil(math.log2(max(L, 2))))
    for i in range(steps):
        stride = 1 << i
        if stride >= L:
            break
        shifted = torch.zeros_like(res)
        shifted[:, stride:] = res[:, :-stride]
        res = res + shifted
    return res


# ── Sharpness regularisation ──────────────────────────────────────────────────

def sharpness_loss(phi: torch.Tensor, strength: float = 0.02) -> torch.Tensor:
    """Penalise oscillators near ±π/2 (undecided state)."""
    return strength * phi.sin().pow(2).mean()


# ── Phase encoder head (real weights, no complex dtype) ───────────────────────

class PhaseEncoderHead(nn.Module):
    """
    Phase encoder using two real weight matrices instead of one complex matrix.
    phi = atan2(E@W_imag.T, E@W_real.T) * tanh(|(E@W_real.T, E@W_imag.T)|) * omega

    Mathematically identical to complex weights (verified to 1e-15 precision).
    Avoids all PyTorch complex dtype issues.
    """

    def __init__(self, D: int, K: int, seed: int = 0, scale: float = 2.0):
        super().__init__()
        self.D = D
        self.K = K
        g = torch.Generator().manual_seed(seed)
        # Initialise from same distribution as complex Rayleigh+uniform
        # Real part: N(0, scale²/2), Imag part: N(0, scale²/2)
        std = scale / math.sqrt(2)
        self.W_real = nn.Parameter(torch.randn(K, D, generator=g) * std)
        self.W_imag = nn.Parameter(torch.randn(K, D, generator=g) * std)
        # Log-uniform frequency bands [0.25, 4.0]
        log_om = torch.empty(K).uniform_(math.log(0.25), math.log(4.0),
                                          generator=g)
        self.omega = nn.Parameter(log_om.exp())

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """E: (..., D) → (..., K) phase."""
        z_r  = E @ self.W_real.T                       # (..., K)
        z_i  = E @ self.W_imag.T                       # (..., K)
        mag  = (z_r.pow(2) + z_i.pow(2)).sqrt() + 1e-12
        gate = mag.tanh()
        phi  = torch.atan2(z_i, z_r) * gate * self.omega
        return phi


# ── Multi-head phase encoder ──────────────────────────────────────────────────

class MultiHeadPhaseEncoder(nn.Module):
    """N_heads independent phase heads → concat → linear projection."""

    def __init__(self, D: int, K_head: int = 256, N_heads: int = 8):
        super().__init__()
        self.N_heads = N_heads
        self.K_head  = K_head
        self.K_total = K_head * N_heads
        self.heads   = nn.ModuleList([
            PhaseEncoderHead(D, K_head, seed=i)
            for i in range(N_heads)
        ])
        self.proj = nn.Linear(self.K_total, self.K_total, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        phi = torch.cat([h(E) for h in self.heads], dim=-1)
        return self.proj(phi)


# ── Phase normalisation ───────────────────────────────────────────────────────

class PhaseNorm(nn.Module):
    """Layer norm via cos/sin decomposition — avoids ±π discontinuity."""

    def __init__(self, K: int, eps: float = 1e-5):
        super().__init__()
        self.K     = K
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(K))
        self.beta  = nn.Parameter(torch.zeros(K))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        c = F.layer_norm(phi.cos(), [self.K], eps=self.eps)
        s = F.layer_norm(phi.sin(), [self.K], eps=self.eps)
        return self.gamma * torch.atan2(s, c) + self.beta


# ── Phase feedforward block ───────────────────────────────────────────────────

class PhaseFFN(nn.Module):
    """
    FFN in phase space. Key Phase 3 addition.

    Phase 2 had 0.07% of params in scan layers → plateau at PPL 294.
    Phase 3 scan layers have 73% of params via this FFN → target PPL <100.

    Input: phi (B, L, K) → decomposes to [cos, sin] → FFN → residual.
    """

    def __init__(self, K: int, expansion: int = 2):
        super().__init__()
        self.w1   = nn.Linear(2 * K, K * expansion)
        self.w2   = nn.Linear(K * expansion, K)
        self.norm = PhaseNorm(K)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.zeros_(self.w2.weight)  # zero init → identity at start
        nn.init.zeros_(self.w2.bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        x = torch.cat([phi.cos(), phi.sin()], dim=-1)
        return self.norm(phi + self.w2(F.gelu(self.w1(x))))


# ── Phase scan layer ──────────────────────────────────────────────────────────

class PhaseScanLayer(nn.Module):
    """
    Causal scan + optional PhaseFFN + residuals.

    Without residuals: gradient explodes ~4.5M× across 4 layers.
    With residuals: ratio ~0.68 (verified).
    """

    def __init__(self, K: int, ffn_expansion: int = 2, use_ffn: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.norm  = PhaseNorm(K)
        self.ffn   = PhaseFFN(K, ffn_expansion) if use_ffn else None

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        ctx        = hillis_steele_scan(phi)
        phi_mixed  = phi + self.alpha.sigmoid() * (ctx - phi)
        phi        = phi + self.norm(phi_mixed)        # residual after scan
        if self.ffn is not None:
            phi = self.ffn(phi)                        # FFN with internal residual
        return phi


# ── Language model ────────────────────────────────────────────────────────────

class PhaseLM(nn.Module):
    """
    Phase-space causal language model.

    IMPORTANT: targets must be PRE-SHIFTED by DataLoader (y = x[1:]).
    No internal shift — avoids double-shift bug (PPL > random).
    """

    def __init__(self,
                 vocab_size:    int   = 8192,
                 D_embed:       int   = 256,
                 K_head:        int   = 256,
                 N_heads:       int   = 8,
                 N_layers:      int   = 4,
                 dropout:       float = 0.1,
                 lam_sharp:     float = 0.02,
                 ffn_expansion: int   = 2,
                 use_ffn:       bool  = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.K_total    = K_head * N_heads
        self.lam_sharp  = lam_sharp

        self.embedding   = nn.Embedding(vocab_size, D_embed)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.encoder     = MultiHeadPhaseEncoder(D_embed, K_head, N_heads)
        self.scan_layers = nn.ModuleList([
            PhaseScanLayer(self.K_total, ffn_expansion, use_ffn)
            for _ in range(N_layers)
        ])
        self.final_norm  = PhaseNorm(self.K_total)
        self.drop        = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lm_head     = nn.Linear(self.K_total, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor,
                targets: torch.Tensor = None):
        E      = self.drop(self.embedding(tokens))
        phi    = self.encoder(E)
        for layer in self.scan_layers:
            phi = layer(phi)
        phi    = self.final_norm(phi)
        phi    = self.drop(phi)
        logits = self.lm_head(phi)

        loss = None
        if targets is not None:
            # targets are PRE-SHIFTED — no internal shift needed
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
            if self.lam_sharp > 0:
                loss = loss + sharpness_loss(phi, self.lam_sharp)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int = 100,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        self.eval()
        tokens = prompt.clone()
        for _ in range(max_new):
            ctx    = tokens[:, -512:]
            logits, _ = self(ctx)
            next_l = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                v, _   = torch.topk(next_l, min(top_k, next_l.size(-1)))
                next_l = next_l.masked_fill(next_l < v[:, [-1]], float('-inf'))
            next_t = torch.multinomial(F.softmax(next_l, dim=-1), 1)
            tokens = torch.cat([tokens, next_t], dim=1)
        return tokens

    def param_count(self) -> dict:
        n = sum(p.numel() for p in self.parameters())
        return {'total': n, 'mb': n * 4 / 1e6}


# ── Intent classifier ─────────────────────────────────────────────────────────

class PhaseIntentClassifier(nn.Module):
    """Phase encoder + hidden layer + softmax for CLINC150 benchmark."""

    def __init__(self, D: int, N_intents: int,
                 K_head: int = 128, N_heads: int = 8,
                 H: int = 512, dropout: float = 0.1,
                 lam_sharp: float = 0.05):
        super().__init__()
        self.K_total   = K_head * N_heads
        self.lam_sharp = lam_sharp
        self.encoder   = MultiHeadPhaseEncoder(D, K_head, N_heads)
        self.drop      = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.hidden    = nn.Linear(self.K_total, H)
        self.cls       = nn.Linear(H, N_intents)

    def forward(self, E: torch.Tensor, labels: torch.Tensor = None):
        phi    = self.drop(self.encoder(E))
        logits = self.cls(self.drop(F.relu(self.hidden(phi))))
        loss   = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if self.lam_sharp > 0:
                loss = loss + sharpness_loss(phi, self.lam_sharp)
        return logits, loss

    def predict(self, E: torch.Tensor) -> torch.Tensor:
        return self.forward(E)[0].argmax(dim=-1)

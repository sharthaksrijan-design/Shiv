//! Phase-SNN Inference Engine
//!
//! CPU inference for the Phase-SNN architecture.
//! Loads weights exported from PyTorch training.
//!
//! Architecture:
//!   MultiHeadPhaseEncoder → PhaseScanLayer × N → LM head
//!
//! Usage:
//!   let model = PhaseLM::load("weights.json").unwrap();
//!   let tokens = vec![2u32, 10, 42, 7];  // <bos> + token ids
//!   let next_logits = model.forward(&tokens);
//!   let next_token = next_logits.argmax();

use ndarray::{Array1, Array2, ArrayView1, Axis, s};
use num_complex::Complex32;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

// ── Weight structures (loaded from JSON export) ───────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct PhaseEncoderHeadWeights {
    /// W: shape [K, D] as flat vec of (real, imag) pairs
    pub w_real: Vec<f32>,
    pub w_imag: Vec<f32>,
    pub omega:  Vec<f32>,
    pub k: usize,
    pub d: usize,
}

#[derive(Serialize, Deserialize)]
pub struct PhaseScanLayerWeights {
    pub alpha:      f32,
    /// PhaseNorm gamma and beta
    pub norm_gamma: Vec<f32>,
    pub norm_beta:  Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelWeights {
    pub vocab_size: usize,
    pub d_embed:    usize,
    pub k_total:    usize,
    pub n_layers:   usize,
    /// Token embedding table: [vocab_size, d_embed]
    pub embedding:  Vec<f32>,
    pub heads:      Vec<PhaseEncoderHeadWeights>,
    pub scan_layers: Vec<PhaseScanLayerWeights>,
    /// LM head: [vocab_size, k_total]
    pub lm_head_w:  Vec<f32>,
    pub lm_head_b:  Vec<f32>,
    /// Vocabulary: token_id → string
    pub vocab:      std::collections::HashMap<u32, String>,
}

// ── Core operations ───────────────────────────────────────────────────────────

/// Phase cosine similarity between two vectors
#[inline]
pub fn phase_cos_sim(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let k = a.len() as f32;
    a.iter().zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).cos())
        .sum::<f32>() / k
}

/// Encode a single embedding through one phase head
/// E: [D], W: [K, D] complex, omega: [K] → phi: [K]
pub fn encode_head(
    e:     &[f32],
    w_real: &[f32],
    w_imag: &[f32],
    omega:  &[f32],
    k: usize, d: usize,
) -> Vec<f32> {
    let mut phi = vec![0.0f32; k];
    for ki in 0..k {
        // z = sum_d E[d] * conj(W[ki, d])
        // conj(W) = (w_real - i*w_imag)
        let mut z_real = 0.0f32;
        let mut z_imag = 0.0f32;
        for di in 0..d {
            let e_d = e[di];
            z_real += e_d * w_real[ki * d + di];
            z_imag += e_d * w_imag[ki * d + di];  // conj flips sign of imag
        }
        let z = Complex32::new(z_real, z_imag);
        let mag  = z.norm() + 1e-12;
        let gate = mag.tanh();
        let angle = z.arg();
        phi[ki] = angle * omega[ki] * gate;
    }
    phi
}

/// Hillis-Steele causal prefix scan on a sequence
/// input: [L, K] → output: [L, K] where output[t] = sum(input[0..=t])
pub fn hillis_steele_scan(x: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let l = x.len();
    if l == 0 { return vec![]; }
    let mut res: Vec<Vec<f32>> = x.to_vec();
    let num_steps = (l as f32).log2().ceil() as usize + 1;
    for i in 0..num_steps {
        let stride = 1usize << i;
        if stride >= l { break; }
        let prev = res.clone();
        for t in stride..l {
            for ki in 0..k {
                res[t][ki] += prev[t - stride][ki];
            }
        }
    }
    res
}

/// PhaseNorm: layer norm applied to cos then sin separately
pub fn phase_norm(phi: &[f32], gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    let k = phi.len();
    let cos_phi: Vec<f32> = phi.iter().map(|&p| p.cos()).collect();
    let sin_phi: Vec<f32> = phi.iter().map(|&p| p.sin()).collect();

    fn layer_norm_1d(x: &[f32]) -> Vec<f32> {
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let var  = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f32>()
                   / x.len() as f32;
        let std  = (var + 1e-5).sqrt();
        x.iter().map(|&xi| (xi - mean) / std).collect()
    }

    let cos_n = layer_norm_1d(&cos_phi);
    let sin_n = layer_norm_1d(&sin_phi);

    // Reconstruct: atan2(sin_norm, cos_norm) * gamma + beta
    (0..k).map(|i| {
        sin_n[i].atan2(cos_n[i]) * gamma[i] + beta[i]
    }).collect()
}

/// One PhaseScanLayer forward pass with residual connection
pub fn scan_layer_forward(
    phi:    &[f32],
    ctx:    &[f32],
    alpha:  f32,
    gamma:  &[f32],
    beta:   &[f32],
) -> Vec<f32> {
    let k = phi.len();
    let alpha_s = 1.0 / (1.0 + (-alpha).exp());  // sigmoid
    // Mix: phi + alpha * (ctx - phi)
    let mixed: Vec<f32> = (0..k).map(|i| {
        phi[i] + alpha_s * (ctx[i] - phi[i])
    }).collect();
    // Normalise
    let normed = phase_norm(&mixed, gamma, beta);
    // Residual: phi + normed
    (0..k).map(|i| phi[i] + normed[i]).collect()
}

// ── Full model ────────────────────────────────────────────────────────────────

pub struct PhaseLM {
    weights: ModelWeights,
}

impl PhaseLM {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let weights: ModelWeights = serde_json::from_str(&json)?;
        println!("Loaded Phase-SNN: vocab={} K={} layers={}",
                 weights.vocab_size, weights.k_total, weights.n_layers);
        Ok(Self { weights })
    }

    /// Forward pass: token ids → logits over vocabulary
    /// tokens: sequence of token ids (including <bos>)
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let w  = &self.weights;
        let l  = tokens.len();
        let d  = w.d_embed;
        let k  = w.k_total;
        let n_heads = w.heads.len();
        let k_head  = k / n_heads;

        // Step 1: embed tokens → [L, D]
        let embeddings: Vec<Vec<f32>> = tokens.iter().map(|&tok| {
            let idx = (tok as usize).min(w.vocab_size - 1);
            w.embedding[idx*d..(idx+1)*d].to_vec()
        }).collect();

        // Step 2: multi-head phase encode → [L, K]
        let mut phi_seq: Vec<Vec<f32>> = embeddings.iter().map(|e| {
            let mut phi_all = Vec::with_capacity(k);
            for head in &w.heads {
                let phi_h = encode_head(e, &head.w_real, &head.w_imag,
                                        &head.omega, head.k, head.d);
                phi_all.extend(phi_h);
            }
            phi_all
        }).collect();

        // Step 3: scan layers with residuals
        for layer in &w.scan_layers {
            let ctx = hillis_steele_scan(&phi_seq, k);
            phi_seq = phi_seq.iter().zip(ctx.iter()).map(|(phi, c)| {
                scan_layer_forward(phi, c, layer.alpha,
                                   &layer.norm_gamma, &layer.norm_beta)
            }).collect();
        }

        // Step 4: LM head on last position → [vocab_size]
        let last_phi = &phi_seq[l - 1];
        let vocab    = w.vocab_size;
        (0..vocab).map(|v| {
            let w_row = &w.lm_head_w[v*k..(v+1)*k];
            let dot: f32 = last_phi.iter().zip(w_row.iter())
                .map(|(&p, &ww)| p * ww).sum();
            dot + w.lm_head_b[v]
        }).collect()
    }

    /// Greedy next-token prediction
    pub fn predict_next(&self, tokens: &[u32]) -> u32 {
        let logits = self.forward(tokens);
        logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(1)
    }

    /// Temperature-sampled generation
    pub fn generate(&self, prompt: &[u32], max_new: usize,
                    temperature: f32) -> Vec<u32> {
        let mut tokens = prompt.to_vec();
        for _ in 0..max_new {
            let logits = self.forward(&tokens);
            // Temperature scaling + softmax
            let max_l  = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scaled: Vec<f32> = logits.iter()
                .map(|&l| ((l - max_l) / temperature).exp()).collect();
            let sum: f32 = scaled.iter().sum();
            let probs: Vec<f32> = scaled.iter().map(|&s| s / sum).collect();
            // Sample
            let r: f32 = rand_f32();
            let mut cumsum = 0.0f32;
            let mut next = 1u32;  // <unk> fallback
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum >= r {
                    next = i as u32;
                    break;
                }
            }
            tokens.push(next);
        }
        tokens
    }
}

// Simple LCG random for no-dependency sampling
static mut RAND_STATE: u64 = 12345;
fn rand_f32() -> f32 {
    unsafe {
        RAND_STATE = RAND_STATE.wrapping_mul(6364136223846793005)
                               .wrapping_add(1442695040888963407);
        ((RAND_STATE >> 33) as f32) / (u32::MAX as f32)
    }
}

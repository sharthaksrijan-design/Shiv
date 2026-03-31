"""
Export Phase-SNN weights from PyTorch to JSON for Rust inference.

Usage (in Colab after training):
    from export_weights import export_model
    export_model(lm2, token_to_id, 'weights.json')
"""

import json
import torch
import numpy as np


def export_model(model, token_to_id: dict, path: str) -> None:
    """
    Export PhaseLM weights to JSON for Rust inference.
    model:        trained PhaseLM instance
    token_to_id:  vocabulary dict
    path:         output JSON file path
    """
    model.eval()
    w = {}

    # Metadata
    w['vocab_size'] = model.vocab_size
    w['d_embed']    = model.embedding.weight.shape[1]
    w['k_total']    = model.encoder.K_total
    w['n_layers']   = len(model.scan_layers)

    # Token embedding
    emb = model.embedding.weight.detach().float().cpu()
    w['embedding'] = emb.numpy().flatten().tolist()

    # Phase encoder heads
    w['heads'] = []
    for head in model.encoder.heads:
        W_c    = head.W.detach().cpu()
        # Resolve conjugate before exporting
        if W_c.is_conj():
            W_c = W_c.resolve_conj()
        W_c    = W_c.to(torch.cfloat)
        omega  = head.omega.detach().float().cpu()
        w['heads'].append({
            'w_real': W_c.real.numpy().flatten().tolist(),
            'w_imag': W_c.imag.numpy().flatten().tolist(),
            'omega':  omega.numpy().tolist(),
            'k':      head.K,
            'd':      head.D,
        })

    # Scan layers
    w['scan_layers'] = []
    for layer in model.scan_layers:
        alpha = float(layer.alpha.detach().cpu())
        gamma = layer.norm.gamma.detach().float().cpu().numpy().tolist()
        beta  = layer.norm.beta.detach().float().cpu().numpy().tolist()
        w['scan_layers'].append({
            'alpha':      alpha,
            'norm_gamma': gamma,
            'norm_beta':  beta,
        })

    # LM head
    lm_w = model.lm_head.weight.detach().float().cpu()
    lm_b = model.lm_head.bias.detach().float().cpu() \
           if model.lm_head.bias is not None \
           else torch.zeros(model.vocab_size)
    w['lm_head_w'] = lm_w.numpy().flatten().tolist()
    w['lm_head_b'] = lm_b.numpy().tolist()

    # Vocabulary
    w['vocab'] = {str(v): k for k, v in token_to_id.items()}

    # Write
    with open(path, 'w') as f:
        json.dump(w, f)

    size_mb = len(json.dumps(w).encode()) / 1e6
    print(f"Exported to {path}  ({size_mb:.1f} MB)")
    print(f"  vocab={w['vocab_size']}  K={w['k_total']}  "
          f"layers={w['n_layers']}  heads={len(w['heads'])}")
    print(f"  Load in Rust: PhaseLM::load(\"{path}\")")

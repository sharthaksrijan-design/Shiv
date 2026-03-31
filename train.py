"""
Phase 3 Training — Multi-Platform, Real Weights
=================================================
Runs on: Kaggle, Colab, Lightning.ai, local GPU/CPU.
Auto-detects environment and configures accordingly.

Key changes from Phase 2:
  - Complex weights → two real matrices (no PyTorch complex issues)
  - PhaseFFN in scan layers (67M new params, was 16K)
  - Platform auto-detection for checkpointing + workers
  - Progress prints on every slow operation
  - Mixed precision (float16) throughout
"""

import os, sys, time, math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# sys.path.insert(0, '/content' if os.path.exists('/content') else '.')

from phase_snn_torch import (PhaseLM, PhaseIntentClassifier,
                              cosine_lr_schedule, sharpness_loss)
from checkpoint import CheckpointManager

# ── Platform detection ────────────────────────────────────────────────────────

def detect_platform():
    """Detect compute environment and return config dict."""
    if os.path.exists('/kaggle/working'):
        return {
            'name':        'Kaggle',
            'ckpt_dir':    '/kaggle/working/phase_snn_ckpts',
            'data_dir':    '/kaggle/working',
            'num_workers': 2,         # Kaggle supports workers
            'pin_memory':  True,
            'persistent':  False,
        }
    elif os.path.exists('/content'):
        return {
            'name':        'Colab',
            'ckpt_dir':    '/content/drive/MyDrive/phase_snn_ckpts',
            'data_dir':    '/content',
            'num_workers': 0,         # Colab forks hang
            'pin_memory':  False,
            'persistent':  False,
        }
    elif os.path.exists('/teamspace'):
        return {
            'name':        'Lightning.ai',
            'ckpt_dir':    '/teamspace/studios/this_studio/phase_snn_ckpts',
            'data_dir':    '/teamspace/studios/this_studio',
            'num_workers': 2,
            'pin_memory':  True,
            'persistent':  False,
        }
    else:
        return {
            'name':        'Local',
            'ckpt_dir':    os.path.expanduser('~/phase_snn_ckpts'),
            'data_dir':    os.path.expanduser('~'),
            'num_workers': 2,
            'pin_memory':  torch.cuda.is_available(),
            'persistent':  False,
        }

PLATFORM = detect_platform()
print(f"Platform: {PLATFORM['name']}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device:   {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"VRAM:     {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Mount Drive on Colab
if PLATFORM['name'] == 'Colab':
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
    except Exception:
        PLATFORM['ckpt_dir'] = '/content/phase_snn_ckpts'
        print("Drive not available — checkpoints saved locally")

os.makedirs(PLATFORM['ckpt_dir'], exist_ok=True)
print(f"Checkpoints: {PLATFORM['ckpt_dir']}")


def clip_grads(parameters, max_norm=1.0):
    """Standard gradient clipping — no complex dtype handling needed."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    total = torch.cat([p.grad.detach().flatten() for p in params]).norm(2)
    clip  = max_norm / (total + 1e-6)
    if clip < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad.detach().mul_(clip)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these
# ═══════════════════════════════════════════════════════════════════════════════

FORCE_FRESH_START  = True    # True = ignore old checkpoints (required for Phase 3)
LM_TOTAL_STEPS     = 30000
CKPT_EVERY         = 500
LM_BATCH           = 16      # per-GPU — safe for all platforms
GRAD_ACCUM         = 4       # effective batch = 64
SEQ_LEN            = 128
VOCAB_SAMPLE       = 10_000  # texts for vocab building
MAX_TRAIN_TOKENS   = 1_000_000
MAX_VAL_TOKENS     =   100_000
LM_LR              = 1e-3
WARMUP             = 1000

# ═══════════════════════════════════════════════════════════════════════════════
# RUN 1: DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RUN 1: DATA")
print("="*60)

from datasets import load_dataset
print("\nLoading WikiText-103...")

# WikiText-103 loading — tries Kaggle local files first, then HuggingFace
# Kaggle dataset: search "vadimkurochkin/wikitext-103" in + Add Data
# Kaggle Internet must be ON for HuggingFace fallback
import os

# Multiple known Kaggle slugs for WikiText-103
WT_KAGGLE_CANDIDATES = [
    '/kaggle/input/wikitext-103',
    '/kaggle/input/wikitext103',
    '/kaggle/input/wikitext-103-raw',
]

def _load_wikitext_local(path):
    """Load WikiText-103 from local Kaggle dataset files."""
    train, val = [], []
    # Try different possible filenames
    for train_name in ['wiki.train.tokens', 'wiki.train.raw',
                       'train.txt', 'wikitext-103/wiki.train.tokens']:
        fpath = os.path.join(path, train_name)
        if os.path.exists(fpath):
            with open(fpath, encoding='utf-8') as f:
                train = [l.strip() for l in f if l.strip()
                         and not l.startswith(' = ')]
            break
    for val_name in ['wiki.valid.tokens', 'wiki.valid.raw',
                     'valid.txt', 'wikitext-103/wiki.valid.tokens']:
        fpath = os.path.join(path, val_name)
        if os.path.exists(fpath):
            with open(fpath, encoding='utf-8') as f:
                val = [l.strip() for l in f if l.strip()
                       and not l.startswith(' = ')]
            break
    return train, val

train_texts, val_texts = [], []
for candidate in WT_KAGGLE_CANDIDATES:
    if os.path.exists(candidate):
        print(f"  Loading WikiText-103 from: {candidate}")
        # List what's inside so user can debug path issues
        try:
            files = os.listdir(candidate)[:6]
            print(f"  Files: {files}")
        except Exception:
            pass
        train_texts, val_texts = _load_wikitext_local(candidate)
        if train_texts:
            print(f"  Loaded {len(train_texts):,} train / {len(val_texts):,} val lines")
            break

if not train_texts:
    print("  WikiText-103 not found locally — downloading via HuggingFace")
    print("  (Kaggle: enable Internet in Notebook Settings)")
    wt103       = load_dataset('wikitext', 'wikitext-103-v1')
    train_texts = [x['text'] for x in wt103['train']      if x['text'].strip()]
    val_texts   = [x['text'] for x in wt103['validation'] if x['text'].strip()]

print(f"  {len(train_texts):,} train / {len(val_texts):,} val articles")

# Vocabulary
print(f"\nBuilding vocabulary ({VOCAB_SAMPLE:,} texts)...")
t0 = time.time()
word_freq = Counter()
for i, text in enumerate(train_texts[:VOCAB_SAMPLE]):
    word_freq.update(text.lower().split())
    if (i+1) % 2000 == 0:
        print(f"  {i+1:,}/{VOCAB_SAMPLE:,} texts, "
              f"{len(word_freq):,} unique words...")

token_to_id = {'<pad>':0, '<unk>':1, '<bos>':2, '<eos>':3}
for word, _ in word_freq.most_common(8192 - 10):
    if word not in token_to_id:
        token_to_id[word] = len(token_to_id)
for c in 'abcdefghijklmnopqrstuvwxyz0123456789.,!?;:()-':
    if c not in token_to_id:
        token_to_id[c] = len(token_to_id)

VOCAB_SIZE  = len(token_to_id)
id_to_token = {v: k for k, v in token_to_id.items()}
BOS_ID = token_to_id['<bos>']
EOS_ID = token_to_id['<eos>']
UNK_ID = token_to_id['<unk>']
print(f"  Vocab: {VOCAB_SIZE:,} tokens  ({time.time()-t0:.1f}s)")

# Coverage check
sample = sum((t.lower().split() for t in val_texts[:500]), [])
cov    = sum(1 for w in sample if w in token_to_id) / max(len(sample), 1)
unk_pct= 1 - cov
print(f"  Val coverage: {cov:.1%}  UNK rate: {unk_pct:.1%}")

def encode(text):
    ids = [BOS_ID]
    ids.extend(token_to_id.get(w, UNK_ID) for w in text.lower().split())
    ids.append(EOS_ID)
    return ids

def decode_tokens(ids):
    return ' '.join(id_to_token.get(i, '') for i in ids
                    if i not in (BOS_ID, EOS_ID, 0))

# Encode
print("\nEncoding corpus (with progress)...")
t0 = time.time()

def encode_corpus(texts, max_tokens, label):
    all_ids = []
    for i, text in enumerate(texts):
        if len(all_ids) >= max_tokens:
            break
        all_ids.extend(encode(text))
        if (i+1) % 5000 == 0:
            print(f"  {label}: {len(all_ids):,} / {max_tokens:,} tokens...")
    return np.array(all_ids[:max_tokens], dtype=np.int32)

train_tokens = encode_corpus(train_texts, MAX_TRAIN_TOKENS, 'train')
val_tokens   = encode_corpus(val_texts,   MAX_VAL_TOKENS,   'val')
print(f"  Train: {len(train_tokens):,}  Val: {len(val_tokens):,}  "
      f"({time.time()-t0:.1f}s)")

class TokenDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data    = torch.tensor(data, dtype=torch.long)
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

train_dl = DataLoader(
    TokenDataset(train_tokens, SEQ_LEN),
    batch_size=LM_BATCH, shuffle=True,
    num_workers=PLATFORM['num_workers'],
    pin_memory=PLATFORM['pin_memory'],
    persistent_workers=False,
)
val_dl = DataLoader(
    TokenDataset(val_tokens, SEQ_LEN),
    batch_size=LM_BATCH, shuffle=False,
    num_workers=PLATFORM['num_workers'],
)

# ═══════════════════════════════════════════════════════════════════════════════
# RUN 2: MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RUN 2: MODEL")
print("="*60)

lm = PhaseLM(
    vocab_size    = VOCAB_SIZE,
    D_embed       = 256,
    K_head        = 256,
    N_heads       = 8,
    N_layers      = 4,
    dropout       = 0.1,
    lam_sharp     = 0.02,
    ffn_expansion = 2,
    use_ffn       = True,
).to(DEVICE)

pc = lm.param_count()
print(f"\nPhase 3 model: K=2048  N_layers=4  FFN_exp=2  vocab={VOCAB_SIZE:,}")
print(f"  Parameters: {pc['total']:,}  ({pc['mb']:.1f}MB)")
print(f"  Real weights only — no complex dtype")

scan_params = sum(p.numel() for l in lm.scan_layers for p in l.parameters())
print(f"  Scan layer params: {scan_params:,} "
      f"({scan_params/pc['total']*100:.1f}% of model)")
print(f"  Phase 2 had: 16,388 (0.07%) — plateau at PPL 294")

# Verify VRAM before training
if DEVICE.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    xb_t = torch.randint(0, VOCAB_SIZE, (LM_BATCH, SEQ_LEN), device=DEVICE)
    yb_t = torch.randint(0, VOCAB_SIZE, (LM_BATCH, SEQ_LEN), device=DEVICE)
    use_amp = True
    with torch.cuda.amp.autocast(enabled=use_amp):
        _, l_t = lm(xb_t, yb_t)
    l_t.backward()
    lm.zero_grad()
    peak = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    del xb_t, yb_t, l_t
    torch.cuda.empty_cache()
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n  VRAM check: {peak:.2f}GB peak / {vram_total:.1f}GB available")
    print(f"  {'✓ safe' if peak < vram_total * 0.7 else '⚠ tight — reduce batch'}")
    use_amp = (DEVICE.type == 'cuda')
else:
    use_amp = False
    print("\n  CPU mode — mixed precision disabled")

# ═══════════════════════════════════════════════════════════════════════════════
# RUN 3: TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RUN 3: TRAINING")
print("="*60)

ckpt_mgr      = CheckpointManager(PLATFORM['ckpt_dir'],
                                   prefix='lm_phase3', keep_last=3)
opt_lm        = optim.AdamW(lm.parameters(), lr=LM_LR,
                              weight_decay=1e-2, betas=(0.9, 0.95))
scaler        = torch.cuda.amp.GradScaler(enabled=use_amp)
start_step    = 1
loss_hist     = []
best_val_loss = float('inf')
best_state    = None

if not FORCE_FRESH_START:
    state = ckpt_mgr.load_latest()
    if state is not None:
        try:
            lm.load_state_dict(state['model'])
            opt_lm.load_state_dict(state['optimizer'])
            start_step    = state['step'] + 1
            loss_hist     = state['loss_hist']
            best_val_loss = state.get('best_val_loss', float('inf'))
            best_state    = state.get('best_state')
            lm.to(DEVICE)
            print(f"  Resumed from step {state['step']}")
        except Exception as e:
            print(f"  Checkpoint incompatible ({e}) — starting fresh")
            start_step = 1
else:
    print("  FORCE_FRESH_START=True — fresh training")

remaining = LM_TOTAL_STEPS - start_step + 1
if remaining <= 0:
    print(f"\n  Done. Increase LM_TOTAL_STEPS or set FORCE_FRESH_START=True.")
else:
    LM_CONFIG = dict(vocab=VOCAB_SIZE, K=2048, layers=4,
                     total_steps=LM_TOTAL_STEPS)
    train_iter = iter(train_dl)
    opt_lm.zero_grad()
    t0 = time.time()

    print(f"\nSteps {start_step}→{LM_TOTAL_STEPS}  "
          f"eff_batch={LM_BATCH*GRAD_ACCUM}  seq={SEQ_LEN}  "
          f"warmup={WARMUP}  amp={use_amp}")
    print(f"\n{'step':>6}  {'loss':>8}  {'ppl':>8}  "
          f"{'val_loss':>9}  {'val_ppl':>8}  {'t':>6}")
    print("-"*56)

    accum = 0.0
    for step in range(start_step, LM_TOTAL_STEPS + 1):
        lm.train()
        cosine_lr_schedule(opt_lm, step, LM_TOTAL_STEPS,
                           LM_LR, lr_min=1e-5, warmup_steps=WARMUP)

        for _ in range(GRAD_ACCUM):
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                xb, yb = next(train_iter)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = lm(xb, yb)
            scaler.scale(loss / GRAD_ACCUM).backward()
            accum += loss.item() / GRAD_ACCUM

        scaler.unscale_(opt_lm)
        clip_grads(lm.parameters())
        scaler.step(opt_lm)
        scaler.update()
        opt_lm.zero_grad()
        loss_hist.append(accum)
        accum = 0.0

        if step % CKPT_EVERY == 0 or step == LM_TOTAL_STEPS:
            lm.eval()
            vl_list = []
            with torch.no_grad():
                for i, (xv, yv) in enumerate(val_dl):
                    if i >= 50: break
                    xv, yv = xv.to(DEVICE), yv.to(DEVICE)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        _, vl = lm(xv, yv)
                    vl_list.append(vl.item())
            val_loss = float(np.mean(vl_list))
            val_ppl  = math.exp(min(val_loss, 20))
            tr_loss  = float(np.mean(loss_hist[-min(100, len(loss_hist)):]))
            tr_ppl   = math.exp(min(tr_loss, 20))

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_state    = {k: v.clone()
                                 for k, v in lm.state_dict().items()}

            mem = ""
            if DEVICE.type == 'cuda' and step <= CKPT_EVERY * 2:
                gb = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
                mem = f"  [{gb:.1f}GB]"

            print(f"{step:>6}  {tr_loss:>8.4f}  {tr_ppl:>8.1f}"
                  f"  {val_loss:>9.4f}  {val_ppl:>8.1f}"
                  f"  {time.time()-t0:>5.0f}s"
                  f"{'  ★' if improved else ''}{mem}")

            ckpt_mgr.save(step, lm, opt_lm, loss_hist, LM_CONFIG,
                          extra={'best_val_loss': best_val_loss,
                                 'best_state': best_state})

    print(f"\n  Best val PPL: {math.exp(min(best_val_loss,20)):.1f}")
    print(f"  Phase 2 best: 294.0")
    print(f"  {'✓ Improvement' if best_val_loss < math.log(294) else '~ Still training'}")

# ═══════════════════════════════════════════════════════════════════════════════
# RUN 4: GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RUN 4: GENERATION")
print("="*60)

if best_state is not None:
    lm.load_state_dict(best_state)
lm.eval()

for prompt in ["the history of", "scientists discovered that",
               "the government announced", "in recent years"]:
    ids  = torch.tensor([encode(prompt)], dtype=torch.long).to(DEVICE)
    out  = lm.generate(ids, max_new=60, temperature=0.8, top_k=50)
    print(f"\n  '{prompt}'")
    print(f"  → '{decode_tokens(out[0].cpu().tolist())}'")

print(f"\n{'='*60}")
print("PHASE 3 COMPLETE")
print(f"{'='*60}")
final_ppl = math.exp(min(best_val_loss, 20))
print(f"  Best val PPL: {final_ppl:.1f}  (Phase 2: 294)")
print(f"  Model: {pc['total']:,} params  ({pc['mb']:.1f}MB)")
print(f"  Platform: {PLATFORM['name']}")

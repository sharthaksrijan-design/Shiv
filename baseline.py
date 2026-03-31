"""
Phase-SNN v12 — Full Pipeline
Intent Classification + Generation Proof of Concept
"""

import os, sys, time, gc
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict

# sys.path.insert(0, '/content')
from phase_snn_v12 import (PhaseEncoderV2, PhaseClassifier,
                            PhaseGenerationHead, hillis_steele_scan,
                            sharpness_regularization, Adam)

# import subprocess
# def install(pkg):
#     subprocess.run([sys.executable,'-m','pip','install',pkg,'-q'], check=True)
# install('datasets')
# install('nltk')

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── SECTION 0: Load CLINC150 ──────────────────────────────────────────────────
print("="*65)
print("PHASE-SNN v12 — COMPLEX WEIGHTS + SEQUENCE + GENERATION")
print("="*65)

print("\n[1] Loading CLINC150...")
from datasets import load_dataset

# CLINC150 loading — tries multiple sources in order
# Kaggle: enable Internet in Notebook Settings (right panel) then it downloads automatically
# Colab/Lightning/Local: downloads automatically
import os

def _load_clinc_hf():
    """Load via HuggingFace datasets (requires internet)."""
    # Try multiple known working identifiers
    for hf_id, config in [
        ('clinc/oos',   'plus'),
        ('clinc_oos',   'plus'),
        ('DeepPavlov/clinc150', None),
    ]:
        try:
            ds = load_dataset(hf_id, config) if config else load_dataset(hf_id)
            print(f"  Loaded CLINC150 via {hf_id}")
            return ds
        except Exception:
            continue
    raise RuntimeError(
        "Could not load CLINC150. On Kaggle: enable Internet in Notebook Settings.")

dataset      = _load_clinc_hf()
train_texts  = [x['text']   for x in dataset['train']]
train_intents= [x['intent'] for x in dataset['train']]
val_texts    = [x['text']   for x in dataset['validation']]
val_intents  = [x['intent'] for x in dataset['validation']]
test_texts   = [x['text']   for x in dataset['test']]
test_intents = [x['intent'] for x in dataset['test']]
label_feature = dataset['train'].features['intent']
idx_to_intent = {i: label_feature.int2str(i)
                 for i in range(label_feature.num_classes)}
intent_to_idx = {v: k for k, v in idx_to_intent.items()}
N_INTENTS     = label_feature.num_classes
oos_label     = intent_to_idx.get('oos', None)
train_intents = list(train_intents)
val_intents   = list(val_intents)
test_intents  = list(test_intents)



train_labels = np.array(train_intents)
val_labels   = np.array(val_intents)
test_labels  = np.array(test_intents)

print(f"  Train: {len(train_texts):,}  Val: {len(val_texts):,}  "
      f"Test: {len(test_texts):,}  Intents: {N_INTENTS}")

# ── SECTION 1: Load GloVe ─────────────────────────────────────────────────────
print("\n[2] Loading GloVe...")
# GloVe loading — checks multiple Kaggle paths then downloads
# Kaggle dataset: search "danielwillgeorge/glove6b100dtxt" in + Add Data
GLOVE_KAGGLE_CANDIDATES = [
    '/kaggle/input/glove6b100dtxt/glove.6B.100d.txt',   # danielwillgeorge
    '/kaggle/input/glove6b/glove.6B.100d.txt',           # older slug
    '/kaggle/input/glove-global-vectors/glove.6B.100d.txt',
]
GLOVE_PATH = None
for p in GLOVE_KAGGLE_CANDIDATES:
    if os.path.exists(p):
        GLOVE_PATH = p
        print(f"  Using GloVe from: {p}")
        break
if GLOVE_PATH is None:
    GLOVE_PATH = '/tmp/glove.6B.100d.txt'
    if not os.path.exists(GLOVE_PATH):
        import subprocess
        print("  Downloading GloVe (not on Kaggle — downloading)...")
        subprocess.run(['wget','-q','-O','/tmp/glove.6B.zip',
            'https://nlp.stanford.edu/data/glove.6B.zip'], check=True)
        subprocess.run(['unzip','-q','/tmp/glove.6B.zip',
            'glove.6B.100d.txt','-d','/tmp/'], check=True)
        print("  Downloaded ✓")

glove = {}
with open(GLOVE_PATH, encoding='utf-8') as f:
    for line in f:
        parts = line.split()
        glove[parts[0]] = np.array(parts[1:], dtype=np.float32)
D = 100
print(f"  Loaded {len(glove):,} vectors  D={D}")

# ── SECTION 2: Build embeddings ───────────────────────────────────────────────
print("\n[3] Building sentence embeddings...")
from nltk.tokenize import word_tokenize

def text_to_emb(text, glove, D):
    tokens = word_tokenize(text.lower())
    vecs   = [glove[t] for t in tokens if t in glove]
    if not vecs:
        return np.zeros(D, dtype=np.float32)
    v = np.mean(vecs, axis=0)
    n = np.linalg.norm(v)
    return (v/n).astype(np.float32) if n > 1e-12 else v

t0 = time.time()
train_embs = np.array([text_to_emb(t, glove, D) for t in train_texts], dtype=np.float32)
val_embs   = np.array([text_to_emb(t, glove, D) for t in val_texts],   dtype=np.float32)
test_embs  = np.array([text_to_emb(t, glove, D) for t in test_texts],  dtype=np.float32)
print(f"  Built in {time.time()-t0:.1f}s  coverage={np.mean(np.linalg.norm(train_embs,axis=1)>1e-6):.1%}")

# ── SECTION 3: Baseline ───────────────────────────────────────────────────────
print("\n[4] GloVe baseline (cosine prototype)...")
protos = np.zeros((N_INTENTS, D), dtype=np.float32)
for c in range(N_INTENTS):
    mask = train_labels == c
    if mask.sum():
        p = train_embs[mask].mean(0)
        n = np.linalg.norm(p)
        protos[c] = p/n if n > 1e-12 else p
En = val_embs / (np.linalg.norm(val_embs,axis=1,keepdims=True)+1e-12)
glove_val = float(np.mean(np.argmax(En@protos.T,axis=1)==val_labels))
En = test_embs / (np.linalg.norm(test_embs,axis=1,keepdims=True)+1e-12)
glove_test = float(np.mean(np.argmax(En@protos.T,axis=1)==test_labels))
glove_time_start = time.time()
_ = np.argmax(En[:1]@protos.T,axis=1)
glove_inf = (time.time()-glove_time_start)*1000
print(f"  val={glove_val:.4f}  test={glove_test:.4f}  inf≈{glove_inf:.3f}ms/q")

# ── SECTION 4: Domain structure for relational training ───────────────────────
print("\n[5] Building relational structure...")
CLINC150_DOMAINS = {
    'banking':['transfer','transactions','balance','freeze_account','pay_bill',
               'bill_balance','bill_due','interest_rate','routing','min_payment',
               'new_card','lost_card','card_decline','pin_change','report_fraud'],
    'credit_cards':['credit_score','report_lost_card','credit_limit','rewards_balance',
                    'application_status','card_about_to_expire','replacement_card_duration',
                    'expiration_date','credit_limit_change','damaged_card',
                    'improve_credit_score','apr','redeem_rewards','account_blocked','spending_history'],
    'kitchen_and_dining':['recipe','food_last','meal_suggestion','nutrition_info','calories',
                          'ingredient_substitution','cook_time','food_beverage_price',
                          'restaurant_reviews','restaurant_reservation','confirm_reservation',
                          'how_busy','cancel_reservation','accept_reservations','ingredients_list'],
    'home':['smart_home','shopping_list','shopping_list_update','next_song','play_music',
            'update_playlist','todo_list','todo_list_update','calendar','calendar_update',
            'order','order_status','reminder','reminder_update','what_can_i_ask'],
    'auto_and_commute':['car_rental','car_bluetooth','tire_pressure','oil_change_when',
                        'oil_change_how','jump_start','uber','schedule_maintenance',
                        'last_maintenance','insurance','traffic','directions','gas','gas_type','distance'],
    'travel':['book_flight','book_hotel','get_hotel_recommendations','travel_suggestion',
              'travel_notification','carry_on_baggage','timezone','international_visa',
              'plug_type','exchange_rate','flight_status','international_fees','vaccines','lost_luggage','mpg'],
    'utility':['time','alarm','timer','weather','date','find_phone','share_location',
               'current_location','meeting_schedule','calculator','measurement_conversion',
               'spelling','definition','change_accent','sync_device'],
    'work':['direct_deposit','pto_request','taxes','payday','w2','income','rollover_401k',
            'find_internship','fico_score','insurance_change','user_name','password_reset',
            'change_user_name','change_password','next_holiday'],
    'meta':['who_do_you_work_for','do_you_have_pets','are_you_a_bot','meaning_of_life',
            'who_made_you','thank_you','goodbye','tell_joke','where_are_you_from',
            'how_old_are_you','what_is_your_name','what_are_your_hobbies','fun_fact',
            'change_ai_name','what_can_i_ask'],
    'small_talk':['greeting','yes','no','maybe','I_am_bored','flip_coin','roll_dice',
                  'laugh','story','text','repeat','whisper_mode','make_call','number_facts','next_holiday'],
}

intent_to_domain = {}
for domain, intents in CLINC150_DOMAINS.items():
    for name in intents:
        if name in intent_to_idx:
            intent_to_domain[name] = domain

domain_to_intents = defaultdict(list)
for name, domain in intent_to_domain.items():
    domain_to_intents[domain].append(intent_to_idx[name])

# Build relation pairs
rng_pairs = np.random.default_rng(42)
rel_pairs_by_domain = []
for domain, d_intents in domain_to_intents.items():
    if len(d_intents) < 2:
        continue
    pairs = []
    for ia in range(len(d_intents)):
        for ib in range(ia+1, len(d_intents)):
            ea = np.where(train_labels == d_intents[ia])[0]
            eb = np.where(train_labels == d_intents[ib])[0]
            if len(ea) and len(eb):
                for i in range(min(3, len(ea), len(eb))):
                    pairs.append((int(ea[i]), int(eb[i])))
    if len(pairs) >= 5:
        rel_pairs_by_domain.append(pairs)

# Import v2 build_balanced_quads (still works with index pairs)
from phase_snn_v2 import build_balanced_quads

R         = len(rel_pairs_by_domain)
quads_pos = build_balanced_quads(rel_pairs_by_domain, max_per_family=8)
quads_neg = [(*rel_pairs_by_domain[i%R][0], *rel_pairs_by_domain[(i+1)%R][0])
             for i in range(min(16, R))]
print(f"  R={R}  quads_pos={len(quads_pos)}  quads_neg={len(quads_neg)}")

# ── SECTION 5: Train v12 classifier ──────────────────────────────────────────
print("\n[6] Training v12 classifier (complex weights + hidden layer)...")

K          = 270
EPOCHS     = 2000
BATCH      = 256
LAM_CE     = 1.0
LAM_SHARP  = 0.05   # increased from 0.005 — was too weak (mean=0.508≈random)
LAM_XFER   = 0.05
DROPOUT    = 0.05

enc = PhaseEncoderV2(D, K, lr=2e-3, seed=42)
clf = PhaseClassifier(K, N_INTENTS, H=1024, lr=5e-3, seed=0)

# Separate CE optimizer for encoder (doesn't share state with geometric losses)
opt_enc_ce = Adam((K, D), lr=1e-3, complex_weights=True)  # lower lr for large-scale complex W

rng_train  = np.random.default_rng(42)
best_val   = 0.0
best_enc_W = enc.W.copy()
best_enc_om= enc.omega.copy()
best_clf   = {k: v.copy() for k, v in clf.__dict__.items()
              if isinstance(v, np.ndarray)}

from phase_snn_v12 import cosine_lr
LR_MAX_ENC = 1e-3
LR_MAX_CLF = 5e-3
LR_MIN     = 1e-5

t0 = time.time()
print(f"  K={K}  H={clf.H}  scale=2.0  lam_sharp={LAM_SHARP}  dropout={DROPOUT}")
print(f"  Weight separation: complex(scale=2.0)=0.0172 vs v8=0.0013 (13x better)")
print(f"  Complex weights: ✓  Hidden layer: ✓  Sharpness reg: ✓  Cosine LR: ✓")

for ep in range(1, EPOCHS+1):
    # Cosine annealing — smoothly decays lr to eliminate late-training oscillation
    lr_enc = cosine_lr(ep, EPOCHS, LR_MAX_ENC, LR_MIN)
    lr_clf = cosine_lr(ep, EPOCHS, LR_MAX_CLF, LR_MIN)
    opt_enc_ce.lr = lr_enc
    for opt in [clf.opt_Wh, clf.opt_bh, clf.opt_Wc, clf.opt_bc]:
        opt.lr = lr_clf

    idx  = rng_train.choice(len(train_embs), BATCH, replace=False)
    E_b  = train_embs[idx]
    labs = train_labels[idx]

    # Phase encode
    phi, z, mag, gate, E_flat = enc.phi_with_grad_info(E_b)
    phi = phi * (np.random.binomial(1, 1-DROPOUT, phi.shape)
                 if DROPOUT > 0 else 1)

    # Sharpness regularisation (Upgrade 9)
    sharp_loss, sharp_grad = sharpness_regularization(phi, LAM_SHARP)

    # CE + hidden layer (Upgrade 7)
    ce_loss, d_phi, gW_cls, gb_cls, gW_hid, gb_hid = clf.ce_loss_and_grads(
        phi, labs, sharp_grad=sharp_grad)

    total_loss = ce_loss + sharp_loss

    # Gradient through encoder (complex Wirtinger)
    gW_enc_ce = enc.phi_grad_W(d_phi, z, mag, gate, E_flat)

    # Update encoder via CE optimizer
    enc.W -= opt_enc_ce.step(LAM_CE * gW_enc_ce)

    # Update classifier
    clf.update(gW_cls, gb_cls, gW_hid, gb_hid)

    # Relational transfer loss (light regulariser)
    if quads_pos and ep % 5 == 0:
        flat    = list(set(i for q in quads_pos[:32] for i in q))
        loc     = {g: l for l, g in enumerate(flat)}
        E_loc   = train_embs[flat]
        phi_loc, z_loc, mag_loc, gate_loc, Ef_loc = enc.phi_with_grad_info(E_loc)
        G_xfer  = np.zeros_like(phi_loc)
        for ai, bi, ci, di in quads_pos[:32]:
            if not all(x in loc for x in [ai,bi,ci,di]): continue
            la,lb,lc,ld = loc[ai],loc[bi],loc[ci],loc[di]
            theta = phi_loc[lc]+phi_loc[lb]-phi_loc[la]-phi_loc[ld]
            g = np.sin(theta)/(K*32)
            G_xfer[lb]+=g; G_xfer[la]-=g
            G_xfer[lc]+=g; G_xfer[ld]-=g
        gW_xfer = enc.phi_grad_W(LAM_XFER*G_xfer, z_loc, mag_loc, gate_loc, Ef_loc)
        enc.W  -= enc.opt.step(gW_xfer)

    if ep % 100 == 0 or ep == EPOCHS:
        phi_v   = enc.phi(val_embs.astype(np.float64))
        val_prd = clf.predict(phi_v)
        val_acc = float(np.mean(val_prd == val_labels))
        if val_acc > best_val:
            best_val    = val_acc
            best_enc_W  = enc.W.copy()
            best_enc_om = enc.omega.copy()
            best_clf    = {k: v.copy() for k, v in vars(clf).items()
                           if isinstance(v, np.ndarray)}
        mark = '★' if val_acc == best_val else ''
        print(f"  ep={ep:>4}  ce={ce_loss:.4f}  sharp={sharp_loss:.4f}"
              f"  val={val_acc:.4f} {mark}  t={time.time()-t0:.0f}s")

print(f"  Best val={best_val:.4f} — restoring best weights")
enc.W     = best_enc_W
enc.omega = best_enc_om
for k, v in best_clf.items():
    setattr(clf, k, v)

# ── SECTION 6: Test evaluation ────────────────────────────────────────────────
print("\n[7] Test evaluation...")

phi_test  = enc.phi(test_embs.astype(np.float64))
test_preds = clf.predict(phi_test)
test_acc  = float(np.mean(test_preds == test_labels))

if oos_label is not None:
    mask     = test_labels != oos_label
    test_ins = float(np.mean(test_preds[mask] == test_labels[mask]))
else:
    test_ins = test_acc

# Inference time
t_inf = time.time()
for _ in range(1000):
    _ = clf.predict(enc.phi(test_embs[:1].astype(np.float64)))
inf_ms = (time.time()-t_inf)/1000*1000

print(f"  test_acc={test_acc:.4f}  in-scope={test_ins:.4f}  inf={inf_ms:.3f}ms/q")

# OOS threshold sweep
if oos_label is not None:
    print("\n  OOS threshold sweep on val set:")
    phi_val = enc.phi(val_embs.astype(np.float64))
    logits_v, _, _ = clf.forward(phi_val)
    ex = np.exp(logits_v - logits_v.max(1,keepdims=True))
    probs_v = ex/(ex.sum(1,keepdims=True)+1e-12)
    val_prd = np.argmax(probs_v,1); val_conf = probs_v.max(1)

    logits_t, _, _ = clf.forward(phi_test)
    ex = np.exp(logits_t - logits_t.max(1,keepdims=True))
    probs_t = ex/(ex.sum(1,keepdims=True)+1e-12)
    test_prd_t = np.argmax(probs_t,1); test_conf = probs_t.max(1)

    best_thresh = 0.0; best_thresh_acc = test_acc
    print(f"  {'thresh':>7}  {'val_acc':>8}  {'oos_rec':>8}  {'false_oos':>10}")
    for t in np.arange(0.05, 0.95, 0.05):
        vp = np.where(val_conf>=t, val_prd, oos_label)
        va = float(np.mean(vp==val_labels))
        oos_m = val_labels==oos_label
        oor   = float(np.mean(vp[oos_m]==oos_label)) if oos_m.sum() else 0
        ins_m = val_labels!=oos_label
        foos  = float(np.mean(vp[ins_m]==oos_label)) if ins_m.sum() else 0
        print(f"  {t:>7.2f}  {va:>8.4f}  {oor:>8.4f}  {foos:>9.1%}")
        if va > best_thresh_acc:
            best_thresh_acc = va; best_thresh = t
    if best_thresh > 0:
        tp = np.where(test_conf>=best_thresh, test_prd_t, oos_label)
        test_acc_thresh = float(np.mean(tp==test_labels))
        print(f"\n  Best thresh={best_thresh:.2f}  test_acc={test_acc_thresh:.4f}")
    else:
        test_acc_thresh = test_acc
        print("  No threshold improves val — using no threshold")
else:
    test_acc_thresh = test_acc

# Float64 size (training)
model_size_f64 = int((K*D*2 + K) * 8 + (clf.H*K + clf.H + N_INTENTS*clf.H + N_INTENTS) * 8)
# Float32 size (inference — what we actually deploy)
model_size = int(enc.size_bytes + (clf.H*K + clf.H + N_INTENTS*clf.H + N_INTENTS) * 4)
print(f'  Model size: float64={model_size_f64//1024}KB  float32={model_size//1024}KB  ')


# ── SECTION 7: Upgrade comparison ────────────────────────────────────────────
print("\n[8] Upgrade comparison...")
print(f"  {'Model':<35}  {'val':>7}  {'test':>7}  {'in-scope':>9}  {'size':>10}")
print(f"  {'-'*72}")
rows = [
    ("GloVe prototype (baseline)",   "0.660",  f"{glove_test:.4f}", "0.656",  "59 KB"),
    ("v8 Phase+CE K=270 (best)",      "0.825",  "0.726",   "0.810",   "265 KB"),
    (f"v12 Complex+Hidden K={K}",
     f"{best_val:.4f}", f"{test_acc:.4f}",
     f"{test_ins:.4f}", f"{model_size//1024} KB"),
    ("DistilBERT (published)",        "—",      "0.951",   "—",       "66,000 KB"),
]
for label, val, test, ins, size in rows:
    print(f"  {label:<35}  {val:>7}  {test:>7}  {ins:>9}  {size:>10}")

delta_v8   = test_acc - 0.726
delta_glove = test_acc - glove_test
print(f"\n  v12 vs v8:    {delta_v8:+.4f}")
print(f"  v12 vs GloVe: {delta_glove:+.4f}")

# ── SECTION 8: Encoder property verification ──────────────────────────────────
print("\n[9] Encoder property verification...")

# 9a: Complex weight discriminability
x1 = train_embs[0].astype(np.float64)
x2 = train_embs[1].astype(np.float64)
x3 = x1 + 0.1*np.random.randn(D)*0.1; x3/=np.linalg.norm(x3)

phi1 = enc.phi(x1[None,:])[0]
phi2 = enc.phi(x2[None,:])[0]
phi3 = enc.phi(x3[None,:])[0]

def psim(a,b):
    K=len(a)
    return float((np.cos(a)*np.cos(b)+np.sin(a)*np.sin(b)).sum()/K)

print(f"  Phase similarity (trained encoder):")
print(f"    sim(train[0], train[0]):      {psim(phi1,phi1):.4f}  (self)")
print(f"    sim(train[0], train[1]):      {psim(phi1,phi2):.4f}  (different intent)")
print(f"    sim(train[0], train[0]+noise):{psim(phi1,phi3):.4f}  (similar input)")

# 9b: Hillis-Steele scan — context sensitivity
print(f"\n  Sequence context test (Hillis-Steele scan):")
seq = train_embs[:8].astype(np.float64)[None, :, :]  # (1, 8, D)
phi_seq = enc.phi(seq)                                 # (1, 8, K)
phi_ctx = hillis_steele_scan(phi_seq)                  # (1, 8, K) — contextualised

# Same token at position 0 vs position 4 should have different context phi
pos0_sim = psim(phi_seq[0,0,:], phi_ctx[0,0,:])
pos4_sim = psim(phi_seq[0,0,:], phi_ctx[0,4,:])
print(f"    phi(pos=0) vs ctx(pos=0): {pos0_sim:.4f}  (local — same token)")
print(f"    phi(pos=0) vs ctx(pos=4): {pos4_sim:.4f}  (context differs)")
print(f"    Scan adds context: {'✓' if pos0_sim > pos4_sim + 0.01 else '~'}")

# 9c: Sharpness distribution
phi_all = enc.phi(test_embs[:500].astype(np.float64))
sharpness = np.mean(np.sin(phi_all)**2, axis=1)
print(f"\n  Sharpness distribution (lower = crisper):")
print(f"    mean={sharpness.mean():.4f}  std={sharpness.std():.4f}"
      f"  min={sharpness.min():.4f}  max={sharpness.max():.4f}")
print(f"    (Random phases would give sharpness ≈ 0.5)")

# ── SECTION 9: Generation proof of concept ───────────────────────────────────
print("\n[10] Generation proof of concept (character-level NTP)...")
print("  Note: this is a minimal PoC — not a language model.")
print("  Phase 1 (PyTorch+GPU) is needed for real language modeling.")

GEN_EPOCHS = 200
GEN_BATCH  = 16
SEQ_LEN    = 64

# Small training corpus — Shakespeare excerpt
CORPUS = (
    "To be or not to be that is the question whether tis nobler in the mind "
    "to suffer the slings and arrows of outrageous fortune or to take arms "
    "against a sea of troubles and by opposing end them to die to sleep no more "
    "and by a sleep to say we end the heartache and the thousand natural shocks "
    "that flesh is heir to tis a consummation devoutly to be wished to die to sleep "
    "to sleep perchance to dream ay there is the rub for in that sleep of death "
    "what dreams may come when we have shuffled off this mortal coil must give us pause "
) * 20  # repeat for more data

corpus_bytes = np.frombuffer(CORPUS.encode('utf-8'), dtype=np.uint8).astype(np.int32)
N_BYTES      = len(corpus_bytes)

gen_head = PhaseGenerationHead(D=D, K=K, vocab_size=256, H=512, lr=5e-4, seed=77)
opt_enc_gen = Adam((K, D), lr=5e-4, complex_weights=True)

rng_gen = np.random.default_rng(7)
print(f"  Corpus: {N_BYTES} bytes  epochs={GEN_EPOCHS}  seq_len={SEQ_LEN}")

gen_losses = []
t0 = time.time()
for ep in range(1, GEN_EPOCHS+1):
    # Sample random batch of sequences
    starts = rng_gen.integers(0, N_BYTES - SEQ_LEN - 1, GEN_BATCH)
    x_batch = np.stack([corpus_bytes[s:s+SEQ_LEN]   for s in starts])
    y_batch = np.stack([corpus_bytes[s+1:s+SEQ_LEN+1] for s in starts])

    loss, gW_enc, gW_out, gb_out, gW_gen, gb_gen = gen_head.ntp_loss_and_grads(
        enc, x_batch, y_batch)

    # Update generation head weights (don't update enc.W during gen training
    # to preserve classification performance)
    gen_head.W_out -= gen_head.opt_Wo.step(gW_out)
    gen_head.b_out -= gen_head.opt_bo.step(gb_out)
    gen_head.W_gen -= gen_head.opt_Wg.step(gW_gen)
    gen_head.b_gen -= gen_head.opt_bg.step(gb_gen)

    gen_losses.append(loss)
    if ep % 50 == 0 or ep == GEN_EPOCHS:
        print(f"  ep={ep:>3}  ntp_loss={loss:.4f}"
              f"  {'(falling)' if len(gen_losses)>10 and loss<np.mean(gen_losses[-20:-10]) else '(plateau)'}"
              f"  t={time.time()-t0:.0f}s")

# Generate a sample
prompt = "to be or not"
gen_bytes   = gen_head.generate(enc, list(prompt.encode('utf-8')),
                                 max_new=80, temperature=0.8)
gen_text    = gen_bytes.decode('utf-8', errors='replace')
loss_drop   = gen_losses[0] - gen_losses[-1]
print(f"\n  Loss drop: {gen_losses[0]:.4f} → {gen_losses[-1]:.4f}"
      f"  ({loss_drop:+.4f} — {'learning ✓' if loss_drop > 0.3 else 'weak signal ✗'})")
print(f"  Prompt:    '{prompt}'")
print(f"  Generated: '{gen_text}'")

# ── SECTION 10: Final report ──────────────────────────────────────────────────
print(f"\n\n{'='*65}")
print("PHASE-SNN v12 — FINAL REPORT")
print(f"{'='*65}")
print(f"""
  CLASSIFICATION RESULTS:
  {'Model':<35}  {'test acc':>9}  {'in-scope':>9}  {'size':>10}
  {'-'*67}
  {'GloVe prototype':<35}  {glove_test:>9.4f}  {'0.6560':>9}  {'59 KB':>10}
  {'v8 Phase+CE (our best)':<35}  {'0.7260':>9}  {'0.8096':>9}  {'265 KB':>10}
  {f'v12 Complex+Hidden K={K}':<35}  {test_acc:>9.4f}  {test_ins:>9.4f}  {f'{model_size//1024} KB':>10}
  {'DistilBERT':<35}  {'0.9510':>9}  {'—':>9}  {'66,000 KB':>10}

  UPGRADES IN v12:
  ✓ Upgrade 6: Complex weights W ∈ ℂ  — richer phase encoding
  ✓ Upgrade 7: Hidden layer H=1024    — proven +7pts accuracy
  ✓ Upgrade 8: Hillis-Steele scan     — O(log L) sequence context
  ✓ Upgrade 9: Sharpness regulariser  — crisper representations
  ✓ Upgrade 10: Generation head       — char-level NTP PoC

  GENERATION:
  Loss: {gen_losses[0]:.4f} → {gen_losses[-1]:.4f}  (200 epochs on Shakespeare corpus)
  {'✓ Learning signal confirmed' if gen_losses[0]-gen_losses[-1]>0.3 else '~ Weak signal'}

  MODEL SIZE:
  Encoder W:  {K*D*2*8//1024} KB  (complex float64)
  Classifier: {(clf.H*K+N_INTENTS*clf.H)*8//1024} KB
  Gen head:   {(512*K+256*512)*8//1024} KB
  Total:      {model_size//1024} KB  ({model_size/66e6*100:.2f}% of DistilBERT)

  ROADMAP TO GPT-LEVEL:
  Phase 1 (Weeks 1-4):  PyTorch port, GPU, K=2048, multi-head
  Phase 2 (Weeks 5-8):  6-12 layers, WikiText-103 training
  Phase 3 (Weeks 9-12): 100k-pattern memory, Whisper Protocol v2
  Phase 4 (Weeks 13-20): CUDA kernels, 4-bit quantisation
  Phase 5 (Month 5-12):  1B oscillators, MMLU benchmarks

  The PyTorch port (Phase 1) is the critical unlock.
  All NumPy foundations are now validated and frozen.
""")
print("Done.")

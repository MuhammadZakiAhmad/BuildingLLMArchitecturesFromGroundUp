"""
train_adapter.py — Attention Calibration Distillation (Phase 2.2 BBH)
Run with: %run -i train_adapter.py
Trains for 10 epochs per run, saves checkpoint, evaluates, then stops.
Re-run the same command to continue training for the next 10 epochs.
"""

import glob
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME   = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DATA_FILE    = "bbh_golden_final_with_conf.json"
CHUNK_EPOCHS = 10
LR           = 1e-3
ADAPTER_RANK = 4
TRAIN_RATIO  = 80   # first 80 records → train, last 16 → test (after seed-42 shuffle)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# SECTION 1 — Models (memory-safe: skip if already loaded)
# =============================================================================
if 'tokenizer' not in globals():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
else:
    print("Tokenizer: already in memory.")

def _load_frozen_model():
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m

if 'teacher_model' not in globals():
    print("Loading teacher model...")
    teacher_model = _load_frozen_model()
else:
    print("Teacher model: already in memory.")

if 'student_model' not in globals():
    print("Loading student model...")
    student_model = _load_frozen_model()
else:
    print("Student model: already in memory.")

# =============================================================================
# SECTION 2 — Token IDs and helpers
# =============================================================================
MC_LETTERS_ALL = ["A", "B", "C", "D", "E", "F"]
MC_TOKEN_IDS   = {
    l: tokenizer.encode(l, add_special_tokens=False)[0]
    for l in MC_LETTERS_ALL
}

def _format(text: str) -> str:
    msgs = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Answer:"

def predict_mc(model, prompt: str, valid_letters: list):
    """Renormalized MC prediction over valid_letters only."""
    inputs = tokenizer(_format(prompt), return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :].float()
        probs  = F.softmax(logits, dim=-1)
    raw   = [probs[0, MC_TOKEN_IDS[l]].item() for l in valid_letters]
    total = sum(raw)
    renorm = [r / total for r in raw]
    best   = renorm.index(max(renorm))
    return valid_letters[best], renorm[best]

# =============================================================================
# SECTION 3 — Dataset (memory-safe, ID-validated)
# =============================================================================
if 'golden_records' not in globals():
    print(f"Loading {DATA_FILE}...")
    with open(DATA_FILE, "r") as f:
        golden_records = json.load(f)
    print(f"  Loaded {len(golden_records)} records.")
else:
    print(f"Dataset: already in memory ({len(golden_records)} records).")

# Always recompute the split (it is deterministic so result is identical each run)
random.seed(42)
_shuffled   = golden_records.copy()
random.shuffle(_shuffled)
TRAIN_TASKS = _shuffled[:TRAIN_RATIO]
TEST_TASKS  = _shuffled[TRAIN_RATIO:]
_train_ids  = set(t["id"] for t in TRAIN_TASKS)
_test_ids   = set(t["id"] for t in TEST_TASKS)
print(f"Split: {len(TRAIN_TASKS)} train / {len(TEST_TASKS)} test")

# =============================================================================
# SECTION 4 — Baseline (measured on PURE student, BEFORE adapter injection)
# =============================================================================
if 'baseline_preds' not in globals() or set(baseline_preds.keys()) != _test_ids:
    print("Running 0-shot baseline on test set (pure student, no adapter)...")
    student_model.eval()
    baseline_preds = {}
    for t in tqdm(TEST_TASKS, desc="Baseline"):
        pred, prob = predict_mc(student_model, t["student_prompt"], t["valid_letters"])
        baseline_preds[t["id"]] = {
            "pred": pred, "prob": round(prob, 4), "correct": pred == t["answer"]
        }
    n_base_correct = sum(1 for v in baseline_preds.values() if v["correct"])
    print(f"Baseline accuracy: {n_base_correct}/{len(TEST_TASKS)}")
else:
    print("Baseline: already cached in memory.")

# =============================================================================
# SECTION 5 — CalibratedAttention architecture
# =============================================================================
def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rope(q, k, cos, sin):
    if cos.dim() == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

class CalibratedAttention(nn.Module):
    """
    Wraps a frozen LlamaAttention and injects:
        S' = S + delta,  delta = (Q_rot @ U_q) @ (K_rot @ U_k)^T
    Only U_q and U_k are trainable.
    """
    def __init__(self, original_attn, rank: int = 4):
        super().__init__()
        self.original_attn    = original_attn
        cfg                   = original_attn.config
        self.num_heads        = cfg.num_attention_heads
        self.num_kv_heads     = cfg.num_key_value_heads
        self.hidden_size      = cfg.hidden_size
        self.head_dim         = self.hidden_size // self.num_heads
        self.num_kv_groups    = self.num_heads // self.num_kv_heads
        self._tuple_len       = None   # determined on first forward

        self.U_q = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, rank))
        self.U_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, rank))
        nn.init.normal_(self.U_q, std=0.02)
        nn.init.normal_(self.U_k, std=0.02)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):

        # Determine expected return-tuple length once
        if self._tuple_len is None:
            with torch.no_grad():
                dummy = self.original_attn(
                    hidden_states=hidden_states, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_value=past_key_value,
                    output_attentions=output_attentions, use_cache=use_cache,
                    cache_position=cache_position, position_embeddings=position_embeddings,
                    **kwargs)
                self._tuple_len = len(dummy) if isinstance(dummy, tuple) else 1

        B, T, _ = hidden_states.shape

        # --- Frozen projections ---
        with torch.no_grad():
            Q = self.original_attn.q_proj(hidden_states)   # (B, T, H*D)
            K = self.original_attn.k_proj(hidden_states)   # (B, T, Hkv*D)
            V = self.original_attn.v_proj(hidden_states)   # (B, T, Hkv*D)

        Q = Q.view(B, T, self.num_heads,    self.head_dim).transpose(1, 2)   # (B,H,T,D)
        K = K.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)   # (B,Hkv,T,D)
        V = V.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)   # (B,Hkv,T,D)

        # --- RoPE (frozen) ---
        with torch.no_grad():
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = self.original_attn.rotary_emb(V, position_ids)
            Q, K = _apply_rope(Q, K, cos.float(), sin.float())

            # GQA expansion
            if self.num_kv_groups > 1:
                K = K.repeat_interleave(self.num_kv_groups, dim=1)   # (B,H,T,D)
                V = V.repeat_interleave(self.num_kv_groups, dim=1)

        # Detach so only U_q/U_k get gradients
        Q = Q.detach().float()
        K = K.detach().float()
        V = V.detach().float()

        # --- Trainable delta ---
        A     = torch.einsum('bhtd,hdr->bhtr', Q, self.U_q)            # (B,H,T,rank)
        Bm    = torch.einsum('bhtd,hdr->bhtr', K, self.U_k)            # (B,H,T,rank)
        delta = torch.matmul(A, Bm.transpose(-1, -2))                  # (B,H,T,T)

        # --- Frozen base scores + causal mask ---
        with torch.no_grad():
            S      = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
            causal = torch.tril(torch.ones(T, T, device=device)).bool()
            S      = S.masked_fill(~causal, torch.finfo(S.dtype).min)

        S_prime = S + delta
        if attention_mask is not None:
            S_prime = S_prime + attention_mask.float()

        attn_weights = F.softmax(S_prime, dim=-1)
        attn_output  = torch.matmul(attn_weights, V)                   # (B,H,T,D)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, self.num_heads * self.head_dim)
        attn_output = attn_output.to(dtype=hidden_states.dtype)
        attn_output = self.original_attn.o_proj(attn_output)

        return (attn_output, None, None, None)[:self._tuple_len]

# =============================================================================
# SECTION 6 — Inject adapter (or reuse existing one via hasattr check)
# =============================================================================
if not hasattr(student_model.model.layers[0].self_attn, 'U_q'):
    trainable_params = []
    for layer in student_model.model.layers:
        cal = CalibratedAttention(layer.self_attn, rank=ADAPTER_RANK).to(device)
        layer.self_attn = cal
        trainable_params.extend([cal.U_q, cal.U_k])
    print("Adapter: freshly injected into student model.")
else:
    trainable_params = []
    for layer in student_model.model.layers:
        trainable_params.extend([layer.self_attn.U_q, layer.self_attn.U_k])
    print("Adapter: already in model — reusing existing layers.")

student_model.eval()   # ensure eval mode after injection
n_trainable = sum(p.numel() for p in trainable_params)
print(f"Trainable scalars: {n_trainable:,}")

# =============================================================================
# SECTION 7 — Pre-compute teacher hidden states (ID-validated cache)
# =============================================================================
_cached_train_ids = globals().get('_cached_train_ids_set', set())
if 'train_pairs' not in globals() or _cached_train_ids != _train_ids:
    print("Pre-computing teacher hidden states (cached after this run)...")
    train_pairs = []
    for t in tqdm(TRAIN_TASKS, desc="Teacher targets"):
        t_inputs = tokenizer(_format(t["teacher_prompt"]), return_tensors="pt").to(device)
        s_inputs = tokenizer(_format(t["student_prompt"]),  return_tensors="pt").to(device)

        with torch.no_grad():
            out_t = teacher_model(**t_inputs, output_hidden_states=True)
            t_hs  = [
                out_t.hidden_states[l][:, -1, :].float().detach()
                for l in range(len(out_t.hidden_states))
            ]

        correct_idx = torch.tensor(
            [t["valid_letters"].index(t["answer"])], dtype=torch.long, device=device
        )
        valid_ids = [MC_TOKEN_IDS[l] for l in t["valid_letters"]]
        train_pairs.append((s_inputs, t_hs, correct_idx, valid_ids))

    _cached_train_ids_set = _train_ids
    n_layers = len(train_pairs[0][1])
    print(f"Cached {len(train_pairs)} pairs | {n_layers} hidden-state layers each.")
else:
    n_layers = len(train_pairs[0][1])
    print(f"Teacher targets: already cached ({len(train_pairs)} pairs, {n_layers} layers).")

# =============================================================================
# SECTION 8 — Gradient check
# =============================================================================
student_model.train()
_test_out = student_model(**train_pairs[0][0], output_hidden_states=False)
_test_logits = _test_out.logits[:, -1, :].float()
_test_mc = _test_logits[:, train_pairs[0][3]]
_test_loss = F.cross_entropy(_test_mc, train_pairs[0][2])
_test_loss.backward()
_grad_ok = all(p.grad is not None and p.grad.abs().sum().item() > 0 for p in trainable_params[:4])
for p in trainable_params:
    if p.grad is not None:
        p.grad.zero_()
student_model.eval()
if _grad_ok:
    print("Gradient check: PASSED — U_q/U_k are in the computation graph.")
else:
    raise RuntimeError("Gradient check FAILED — adapter is not connected to the graph!")

# =============================================================================
# SECTION 9 — Checkpoint loading and training setup
# =============================================================================
optimizer   = optim.AdamW(trainable_params, lr=LR, weight_decay=0.0)
start_epoch = 1

ckpt_files = sorted(glob.glob("adapter_epoch_*.pt"))
if ckpt_files:
    latest = ckpt_files[-1]
    try:
        epoch_num   = int(latest.split("adapter_epoch_")[1].split(".pt")[0])
        start_epoch = epoch_num + 1
        ckpt        = torch.load(latest, map_location=device)
        for p, saved in zip(trainable_params, ckpt["adapter_weights"]):
            p.data.copy_(saved.to(device))
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"Resumed from {latest} — starting at Epoch {start_epoch}.")
    except Exception as e:
        print(f"Checkpoint load failed ({e}) — starting fresh from Epoch 1.")
        start_epoch = 1
else:
    print("No checkpoint found — starting fresh from Epoch 1.")

target_epoch = start_epoch + CHUNK_EPOCHS - 1
print(f"\n{'='*60}")
print(f"Training Epochs {start_epoch} → {target_epoch}")
print(f"Loss = 1.0×CE  +  0.1×LayerCosine({n_layers} layers)  +  0.001×L2")
print(f"{'='*60}\n")

# =============================================================================
# SECTION 10 — Training loop
# =============================================================================
for epoch in range(start_epoch, target_epoch + 1):
    student_model.train()
    optimizer.zero_grad()

    total_loss = total_ce = total_cos = total_l2 = 0.0

    for s_inputs, t_hs, correct_idx, valid_ids in train_pairs:
        out = student_model(**s_inputs, output_hidden_states=True)

        # Task CE loss (renormalized over valid tokens only)
        mc_logits = out.logits[:, -1, :].float()[:, valid_ids]
        ce = F.cross_entropy(mc_logits, correct_idx)

        # Layer-wise cosine distillation (all 25 layers)
        cos_loss = sum(
            1.0 - F.cosine_similarity(
                out.hidden_states[l][:, -1, :].float(),
                t_hs[l], dim=-1
            ).mean()
            for l in range(n_layers)
        ) / n_layers

        loss = 1.0 * ce + 0.1 * cos_loss
        loss.backward()

        total_loss += loss.item()
        total_ce   += ce.item()
        total_cos  += cos_loss.item()

    # L2 regularization added once per epoch (not per sample to avoid double-counting)
    l2 = sum(p.pow(2).sum() for p in trainable_params)
    (0.001 * l2).backward()
    total_l2 += l2.item()

    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()

    N = len(train_pairs)
    print(f"Epoch {epoch:03d} | Loss: {total_loss/N:.4f}  "
          f"CE: {total_ce/N:.4f}  Cos: {total_cos/N:.4f}  L2: {total_l2:.1f}")

# =============================================================================
# SECTION 11 — Save checkpoint
# =============================================================================
save_path = f"adapter_epoch_{target_epoch:03d}.pt"
torch.save({
    "epoch":           target_epoch,
    "adapter_weights": [p.data.cpu() for p in trainable_params],
    "optimizer_state": optimizer.state_dict(),
}, save_path)
print(f"\nCheckpoint saved → {save_path}")

# =============================================================================
# SECTION 12 — Evaluation
# =============================================================================
student_model.eval()
after_correct = flipped = regressed = 0

print(f"\n{'='*70}")
print(f"EVALUATION AT EPOCH {target_epoch}")
print(f"{'='*70}")
print(f"{'ID':<6} {'Task':<34} {'Ans':>3}  {'Base':>5}  {'Now':>5}  Change")
print("-" * 70)

for t in TEST_TASKS:
    base         = baseline_preds[t["id"]]
    pred, prob   = predict_mc(student_model, t["student_prompt"], t["valid_letters"])
    correct      = pred == t["answer"]
    after_correct += correct

    b_mark = "✅" if base["correct"] else "❌"
    a_mark = "✅" if correct        else "❌"

    if not base["correct"] and correct:
        change   = "← RECALIBRATED ✨"
        flipped += 1
    elif base["correct"] and not correct:
        change     = "← REGRESSED ⚠️"
        regressed += 1
    else:
        change = ""

    print(f"{t['id']:<6} {t['task']:<34} {t['answer']:>3}  "
          f"{base['pred']} {b_mark}  {pred} {a_mark}  {change}")

print("-" * 70)
base_n = sum(1 for v in baseline_preds.values() if v["correct"])
total  = len(TEST_TASKS)
print(f"\nBaseline (0-shot)       : {base_n}/{total} ({base_n/total:.0%})")
print(f"Epoch {target_epoch:<3} (0-shot)       : {after_correct}/{total} ({after_correct/total:.0%})")
print(f"Recalibrated  (❌ → ✅) : +{flipped}")
if regressed:
    print(f"Regressed     (✅ → ❌) : -{regressed}")
print(f"\nDone. Run '%run -i train_adapter.py' again to train Epochs "
      f"{target_epoch+1}–{target_epoch+CHUNK_EPOCHS}.")

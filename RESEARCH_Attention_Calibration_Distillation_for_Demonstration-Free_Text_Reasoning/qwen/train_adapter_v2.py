"""
train_adapter_v2.py
Distills pre-computed teacher hidden states into the 0-shot student model.
Features sequential loading (Teacher completely purged from memory).
"""
import glob
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# =============================================================================
# PURGE VRAM
# =============================================================================
print("Purging leftover models from VRAM...")
for var in ['teacher_model', 'model']:
    if var in globals():
        del globals()[var]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME   = "Qwen/Qwen2.5-3B-Instruct"
DATA_FILE    = "bbh_golden_v2.json"
CHUNK_EPOCHS = 10
LR           = 1e-3
ADAPTER_RANK = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# SECTION 1 — Models & Tokenizer
# =============================================================================
if 'tokenizer' not in globals():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
else:
    print("Tokenizer already loaded.")

if 'student_model' not in globals():
    print("Loading student model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    for p in student_model.parameters():
        p.requires_grad = False
    student_model.eval()
else:
    print("Student model already loaded.")

# =============================================================================
# SECTION 2 — Dataset & Helpers
# =============================================================================
MC_LETTERS_ALL = ["A", "B", "C", "D", "E", "F"]
MC_TOKEN_IDS   = {
    l: tokenizer.encode(l, add_special_tokens=False)[-1]
    for l in MC_LETTERS_ALL
}

def _format(text: str) -> str:
    msgs = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Answer:"

def predict_mc(model, prompt: str, valid_letters: list):
    inputs = tokenizer(_format(prompt), return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :].float()
        probs  = F.softmax(logits, dim=-1)
    raw   = [probs[0, MC_TOKEN_IDS[l]].item() for l in valid_letters]
    total = sum(raw)
    if total == 0:
        return valid_letters[0], 0.0
    renorm = [r / total for r in raw]
    best   = renorm.index(max(renorm))
    return valid_letters[best], renorm[best]

with open(DATA_FILE, "r") as f:
    golden_records = json.load(f)

random.seed(42)
_shuffled   = golden_records.copy()
random.shuffle(_shuffled)

# DYNAMIC 80/20 SPLIT
train_ratio_idx = int(len(_shuffled) * 0.8)
TRAIN_TASKS = _shuffled[:train_ratio_idx]
TEST_TASKS  = _shuffled[train_ratio_idx:]
print(f"Dynamic Split: {len(TRAIN_TASKS)} train / {len(TEST_TASKS)} test")

# =============================================================================
# SECTION 3 — Baseline Evaluation (Pure Base Model, Before Adapter!)
# =============================================================================
if 'baseline_preds' not in globals() or len(baseline_preds) != len(TEST_TASKS):
    print("Running 0-shot (Student) and 2-shot (Teacher) baselines on test set...")
    student_model.eval()
    baseline_preds = {}
    for t in tqdm(TEST_TASKS, desc="Baseline"):
        # 0-shot base
        s_pred, s_prob = predict_mc(student_model, t["student_prompt"], t["valid_letters"])
        # 2-shot base (Since teacher and student are the same model, we just use the 2-shot prompt!)
        t_pred, t_prob = predict_mc(student_model, t["teacher_prompt"], t["valid_letters"])
        
        baseline_preds[t["id"]] = {
            "s_pred": s_pred, "s_correct": s_pred == t["answer"],
            "t_pred": t_pred, "t_correct": t_pred == t["answer"]
        }
else:
    print("Baseline already cached.")

# =============================================================================
# SECTION 4 — Load Cached Teacher States
# =============================================================================
print("Loading pre-computed teacher states from disk...")
train_pairs_cpu = torch.load("teacher_hidden_states.pt")
train_pairs = []
for s_inputs_cpu, t_hs_cpu, correct_idx, valid_ids, t_id in train_pairs_cpu:
    s_inputs = {k: v.to(device) for k, v in s_inputs_cpu.items()}
    t_hs = [h.to(device) for h in t_hs_cpu]
    correct_idx = correct_idx.to(device)
    train_pairs.append((s_inputs, t_hs, correct_idx, valid_ids))
n_layers = len(train_pairs[0][1])
print(f"Loaded {len(train_pairs)} pairs with {n_layers} layers into VRAM.")

# =============================================================================
# SECTION 5 — CalibratedAttention Architecture
# =============================================================================
def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rope(q, k, cos, sin):
    if cos.dim() == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

class CalibratedAttention(nn.Module):
    def __init__(self, original_attn, rank: int = 4):
        super().__init__()
        self.original_attn    = original_attn
        cfg                   = original_attn.config
        self.num_heads        = cfg.num_attention_heads
        self.num_kv_heads     = getattr(cfg, "num_key_value_heads", self.num_heads)
        self.hidden_size      = cfg.hidden_size
        self.head_dim         = self.hidden_size // self.num_heads
        self.num_kv_groups    = self.num_heads // self.num_kv_heads
        self._tuple_len       = None

        self.U_q = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, rank))
        self.U_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, rank))
        nn.init.normal_(self.U_q, std=0.02)
        nn.init.normal_(self.U_k, std=0.02)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):

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

        with torch.no_grad():
            Q = self.original_attn.q_proj(hidden_states)
            K = self.original_attn.k_proj(hidden_states)
            V = self.original_attn.v_proj(hidden_states)

        Q = Q.view(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        with torch.no_grad():
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = self.original_attn.rotary_emb(V, position_ids)
            Q, K = _apply_rope(Q, K, cos.float(), sin.float())

            if self.num_kv_groups > 1:
                K = K.repeat_interleave(self.num_kv_groups, dim=1)
                V = V.repeat_interleave(self.num_kv_groups, dim=1)

        Q = Q.detach().float()
        K = K.detach().float()
        V = V.detach().float()

        A     = torch.einsum('bhtd,hdr->bhtr', Q, self.U_q)
        Bm    = torch.einsum('bhtd,hdr->bhtr', K, self.U_k)
        delta = torch.matmul(A, Bm.transpose(-1, -2))

        with torch.no_grad():
            S      = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
            causal = torch.tril(torch.ones(T, T, device=device)).bool()
            S      = S.masked_fill(~causal, torch.finfo(S.dtype).min)

        S_prime = S + delta
        if attention_mask is not None:
            S_prime = S_prime + attention_mask.float()

        attn_weights = F.softmax(S_prime, dim=-1)
        attn_output  = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, self.num_heads * self.head_dim)
        attn_output = attn_output.to(dtype=hidden_states.dtype)
        attn_output = self.original_attn.o_proj(attn_output)

        return (attn_output, None, None, None)[:self._tuple_len]

# =============================================================================
# SECTION 6 — Inject Adapter
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

student_model.eval()
print(f"Trainable scalars: {sum(p.numel() for p in trainable_params):,}")

# =============================================================================
# SECTION 7 — Checkpoint loading
# =============================================================================
optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=0.0)
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
print(f"\nTraining Epochs {start_epoch} → {target_epoch}")

# =============================================================================
# SECTION 8 — Training loop
# =============================================================================
for epoch in range(start_epoch, target_epoch + 1):
    student_model.train()
    optimizer.zero_grad()

    total_loss = total_ce = total_cos = total_l2 = 0.0

    for s_inputs, t_hs, correct_idx, valid_ids in train_pairs:
        out = student_model(**s_inputs, output_hidden_states=True)

        mc_logits = out.logits[:, -1, :].float()[:, valid_ids]
        ce = F.cross_entropy(mc_logits, correct_idx)

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

    l2 = sum(p.pow(2).sum() for p in trainable_params)
    (0.001 * l2).backward()
    total_l2 += l2.item()

    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()

    N = len(train_pairs)
    print(f"Epoch {epoch:03d} | Loss: {total_loss/N:.4f}  "
          f"CE: {total_ce/N:.4f}  Cos: {total_cos/N:.4f}  L2: {total_l2:.1f}")

save_path = f"adapter_epoch_{target_epoch:03d}.pt"
torch.save({
    "epoch":           target_epoch,
    "adapter_weights": [p.data.cpu() for p in trainable_params],
    "optimizer_state": optimizer.state_dict(),
}, save_path)
print(f"\nCheckpoint saved → {save_path}")

# =============================================================================
# SECTION 9 — Evaluation
# =============================================================================
student_model.eval()
after_correct = 0

print(f"\n{'='*90}")
print(f"EVALUATION AT EPOCH {target_epoch}")
print(f"{'='*90}")
print(f"{'ID':<6} {'Task':<30} {'Ans':>3}  | {'Tch(2S)':>7} | {'Base(0S)':>8} | {'Now(0S)':>8}")
print("-" * 90)

for t in TEST_TASKS:
    base = baseline_preds[t["id"]]
    pred, prob = predict_mc(student_model, t["student_prompt"], t["valid_letters"])
    now_corr = (pred == t["answer"])
    after_correct += now_corr

    t_mark = "✅" if base["t_correct"] else "❌"
    b_mark = "✅" if base["s_correct"] else "❌"
    n_mark = "✅" if now_corr          else "❌"

    change = ""
    if not base["s_correct"] and now_corr:
        change = "← RECALIBRATED ✨"
    elif base["s_correct"] and not now_corr:
        change = "← REGRESSED ⚠️"

    print(f"{t['id']:<6} {t['task'][:28]:<30} {t['answer']:>3}  | "
          f"{base['t_pred']} {t_mark}   | "
          f"{base['s_pred']} {b_mark}    | "
          f"{pred} {n_mark}    {change}")

print("-" * 90)
base_n = sum(1 for v in baseline_preds.values() if v["s_correct"])
tch_n  = sum(1 for v in baseline_preds.values() if v["t_correct"])
total  = len(TEST_TASKS)

print(f"\nTeacher (2-shot)      : {tch_n}/{total} ({tch_n/total:.0%})")
print(f"Baseline (0-shot)     : {base_n}/{total} ({base_n/total:.0%})")
print(f"Epoch {target_epoch:<3} (0-shot)     : {after_correct}/{total} ({after_correct/total:.0%})")

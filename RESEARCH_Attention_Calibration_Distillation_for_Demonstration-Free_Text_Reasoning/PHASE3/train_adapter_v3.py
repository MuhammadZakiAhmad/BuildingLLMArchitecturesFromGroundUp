"""
train_adapter_v3.py

Phase 3 Implementation: Hyper-ICL Architecture for CoT Distillation
Features:
- Fix 1: Full Sequence Distillation (Aligns all query tokens, not just the last)
- Fix 2: Bypasses Frozen Embedding Layer during distillation
- Fix 3: Correct KV Cache Updates for autoregressive generation (.generate() support)
- Fix 4: Logit-level Intervention with variance scaling (1 / sqrt(r))
- Fix 5: Safe Causal Masking (Mask applied after delta injection)
- Hyper-ICL: Query-Adaptive Token-Wise Modulation (Gate g_{l,h})
- Hyper-ICL: Layer-wise Hyperbolic Anchor Distillation Loss (Lorentz Geodesic)
- Hyper-ICL: Ultra-Sparse Parameter Interventions (Targeting specific layers)
"""
import glob
import json
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# =============================================================================
# CONFIG & HYPERPARAMETERS
# =============================================================================
MODEL_NAME    = "Qwen/Qwen2.5-3B-Instruct"
DATA_FILE     = "pure_golden_records.json"  # Adjusted path for PHASE3 folder
TEACHER_CACHE = "teacher_hidden_states_phase3.pt"
CHUNK_EPOCHS  = 10
LR            = 2e-4  # Lowered from 1e-3 for stability
GRAD_ACCUM    = 8     # Accumulate gradients over 8 examples
ADAPTER_RANK  = 32
TARGET_LAYERS = [-4, -3, -2, -1] # Ultra-Sparse: Only apply adapter to last 4 layers
CURVATURE_K   = 0.1  # Kappa for Hyperbolic space (kappa > 0)
LAMBDA_SUP    = 0.5  # Weight for task supervision vs distillation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# SECTION 1 — Models & Tokenizer
# =============================================================================
print("Checking for existing models in memory...")
if 'tokenizer' not in globals():
    print("Loading tokenizer natively...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
else:
    print("Tokenizer found in memory!")

if 'student_model' not in globals():
    print("Loading student model natively...")
    student_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    for p in student_model.parameters():
        p.requires_grad = False
    student_model.eval()
else:
    print("Student model found in memory!")
    # Ensure it's in eval mode and frozen (if it was previously modified)
    for p in student_model.parameters():
        p.requires_grad = False
    student_model.eval()

# =============================================================================
# SECTION 2 — Dataset Parsing
# =============================================================================
# Load the 90 golden records from JSON
try:
    with open(DATA_FILE, "r") as f:
        golden_records = json.load(f)
    print(f"Loaded {len(golden_records)} golden records from {DATA_FILE}")
except Exception as e:
    print(f"Warning: Could not load dataset {DATA_FILE} - {e}")
    print("Please upload the dataset or adjust the path.")
    golden_records = []

# =============================================================================
# SECTION 3 — Hyperbolic Loss & Calibrated Attention Architecture
# =============================================================================

def lorentz_geodesic_loss(student_h, teacher_h, kappa=0.1):
    """
    Computes the Layer-wise Hyperbolic Anchor Distillation loss.
    student_h, teacher_h: (B, T, D)
    """
    B, T, D = student_h.shape
    
    # Equation 12: Layer Norm
    u_s = F.layer_norm(student_h.float(), (D,))
    u_t = F.layer_norm(teacher_h.float(), (D,))
    
    # Equation 13: Lorentz Exponential Map
    o = torch.zeros(D + 1, device=device)
    o[0] = math.sqrt(1.0 / kappa)
    
    def exp_map(u):
        # Pad to (B, T, D+1)
        u_bar = torch.cat([torch.zeros(B, T, 1, device=device), u], dim=-1)
        norm_u = torch.norm(u, p=2, dim=-1, keepdim=True)
        safe_norm = torch.clamp(norm_u, min=1e-6)  # Avoid div by zero
        
        sqrt_k = math.sqrt(kappa)
        cosh_term = torch.cosh(sqrt_k * norm_u)
        sinh_term = torch.sinh(sqrt_k * safe_norm) / (sqrt_k * safe_norm)
        
        P = cosh_term * o.view(1, 1, -1) + sinh_term * u_bar
        return P
        
    P_s = exp_map(u_s)
    P_t = exp_map(u_t)
    
    # Equation 10: Lorentzian inner product
    inner_L = -P_s[..., 0] * P_t[..., 0] + (P_s[..., 1:] * P_t[..., 1:]).sum(dim=-1)
    
    # Equation 15: Lorentz geodesic distance
    arg = torch.clamp(-kappa * inner_L, min=1.0 + 1e-6) # Clamp for arcosh stability
    dist = math.sqrt(1.0 / kappa) * torch.acosh(arg)
    
    return (dist ** 2).mean()

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
        self.rank             = rank

        # Low-rank intervention matrices
        self.U_q = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, rank))
        self.U_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, rank))
        nn.init.normal_(self.U_q, std=0.02)
        nn.init.normal_(self.U_k, std=0.02)
        
        # Query-Adaptive Modulation Gate parameters (Equation 9)
        self.gate_w = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.gate_b = nn.Parameter(torch.zeros(self.num_heads, 1))
        nn.init.normal_(self.gate_w, std=0.02)

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

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        with torch.no_grad():
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = self.original_attn.rotary_emb(V, position_ids)
            Q, K = _apply_rope(Q, K, cos.float(), sin.float())
            
        # [FIX 3]: KV Cache handling for Autoregressive Generation
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                # New HF Cache object API
                layer_idx = getattr(self.original_attn, "layer_idx", 0)
                cache_kwargs = {"cache_position": cache_position}
                K, V = past_key_value.update(K, V, layer_idx, cache_kwargs)
            else:
                # Legacy tuple API
                past_k, past_v = past_key_value
                K = torch.cat([past_k, K], dim=-2)
                V = torch.cat([past_v, V], dim=-2)
                past_key_value = (K, V)
        
        T_kv = K.shape[-2]

        with torch.no_grad():
            if self.num_kv_groups > 1:
                K_expanded = K.repeat_interleave(self.num_kv_groups, dim=1)
                V_expanded = V.repeat_interleave(self.num_kv_groups, dim=1)
            else:
                K_expanded = K
                V_expanded = V

        Q = Q.detach().float()
        K_expanded = K_expanded.detach().float()
        V_expanded = V_expanded.detach().float()

        # [FIX 4]: Variance Scaling on logit intervention
        A = torch.einsum('bhtd,hdr->bhtr', Q, self.U_q)
        Bm = torch.einsum('bhtd,hdr->bhtr', K_expanded, self.U_k)
        delta = torch.matmul(A, Bm.transpose(-1, -2)) / math.sqrt(self.rank)

        # [Hyper-ICL]: Query-Adaptive Token-Wise Modulation
        Q_norm = F.layer_norm(Q, (self.head_dim,))
        w = self.gate_w.view(1, self.num_heads, 1, self.head_dim)
        b = self.gate_b.view(1, self.num_heads, 1)
        g = torch.sigmoid((Q_norm * w).sum(dim=-1) + b) # (B, H, T)
        delta = delta * g.unsqueeze(-1) # Scale row-wise by query gate

        # [FIX 5]: Safe Causal Masking (Compute S_prime, then mask)
        with torch.no_grad():
            S = torch.matmul(Q, K_expanded.transpose(-1, -2)) / math.sqrt(self.head_dim)
            
        S_prime = S + delta

        if attention_mask is not None:
            # HF attention_mask already handles causal masking safely
            S_prime = S_prime + attention_mask.float()
        else:
            if T > 1:
                causal = torch.tril(torch.ones(T, T_kv, device=device)).bool()
                if T_kv > T:
                    causal = torch.cat([torch.ones(T, T_kv - T, device=device, dtype=torch.bool), causal], dim=-1)
                S_prime = S_prime.masked_fill(~causal, torch.finfo(S_prime.dtype).min)

        attn_weights = F.softmax(S_prime, dim=-1)
        attn_output  = torch.matmul(attn_weights, V_expanded)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, self.num_heads * self.head_dim)
        attn_output = attn_output.to(dtype=hidden_states.dtype)
        attn_output = self.original_attn.o_proj(attn_output)

        ret = (attn_output, attn_weights if output_attentions else None, past_key_value)
        return ret[:self._tuple_len]

# =============================================================================
# SECTION 4 — Inject Adapter (Ultra-Sparse)
# =============================================================================
trainable_params = []
injected_layers = []

# We map TARGET_LAYERS (e.g., [-4, -3, -2, -1]) to actual layer indices
total_layers = len(student_model.model.layers)
actual_target_layers = [l if l >= 0 else total_layers + l for l in TARGET_LAYERS]

for i, layer in enumerate(student_model.model.layers):
    if i in actual_target_layers:
        if not hasattr(layer.self_attn, 'U_q'):
            cal = CalibratedAttention(layer.self_attn, rank=ADAPTER_RANK).to(device)
            layer.self_attn = cal
            print(f"Injected adapter into layer {i}")
        # Always ensure the adapter parameters require gradients (fixes reload freezing bug)
        layer.self_attn.U_q.requires_grad = True
        layer.self_attn.U_k.requires_grad = True
        layer.self_attn.gate_w.requires_grad = True
        layer.self_attn.gate_b.requires_grad = True
        
        trainable_params.extend([
            layer.self_attn.U_q, layer.self_attn.U_k, 
            layer.self_attn.gate_w, layer.self_attn.gate_b
        ])
        injected_layers.append(i)

print(f"Ultra-Sparse Injection complete. Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

# =============================================================================
# SECTION 5 — Training Loop (Template)
# =============================================================================
# NOTE: To run this, you will need to load `train_pairs` from `teacher_hidden_states_phase3.pt`.
# Make sure to upload the dataset and cache to your Colab `/content` folder!
try:
    print(f"Attempting to load {TEACHER_CACHE}...")
    train_pairs_cpu = torch.load(TEACHER_CACHE)
    train_pairs = []
    for s_inputs_cpu, t_hs_cpu, correct_idx, valid_ids in train_pairs_cpu:
        s_inputs = {k: v.to(device) for k, v in s_inputs_cpu.items()}
        t_hs = {k: v.to(device) for k, v in t_hs_cpu.items()}
        train_pairs.append((s_inputs, t_hs, correct_idx, valid_ids))
    print(f"Loaded {len(train_pairs)} pairs.")
except Exception as e:
    print(f"Could not load teacher cache: {e}")
    print("Please run the teacher caching script first and upload the .pt file.")
    train_pairs = []

if train_pairs:
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
    
    for epoch in range(start_epoch, target_epoch + 1):
        student_model.train()
        total_loss = total_ce = total_hyp = 0.0
        
        optimizer.zero_grad()
        for step, (s_inputs, t_hs, correct_idx, valid_ids) in enumerate(train_pairs):
            
            out = student_model(**s_inputs, output_hidden_states=True)

            # Supervision Loss (Optional for CoT distillation, but good for grounding)
            mc_logits = out.logits[:, -1, :].float()[:, valid_ids]
            ce = F.cross_entropy(mc_logits, torch.tensor([correct_idx], device=device))

            # [FIX 1 & 2]: Full sequence distillation on specific layers
            hyperbolic_loss = 0.0
            
            for layer_idx in actual_target_layers:
                # Student transformer layer outputs start at index 1 in out.hidden_states
                s_h = out.hidden_states[layer_idx + 1]
                t_h = t_hs[layer_idx] # t_hs is the populated dictionary
                
                # Align the trailing tokens (the query part)
                align_len = min(s_h.shape[1], t_h.shape[1])
                s_h_aligned = s_h[:, -align_len:, :]
                t_h_aligned = t_h[:, -align_len:, :]
                
                hyperbolic_loss += lorentz_geodesic_loss(s_h_aligned, t_h_aligned, kappa=CURVATURE_K)
                
            hyperbolic_loss = hyperbolic_loss / len(actual_target_layers)

            # Combine Losses and scale by accumulation steps
            loss = LAMBDA_SUP * ce + hyperbolic_loss
            loss_scaled = loss / GRAD_ACCUM
            loss_scaled.backward()
            
            # Step optimizer only after accumulating gradients
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_pairs):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_ce   += ce.item()
            total_hyp  += hyperbolic_loss.item()
            
        N = len(train_pairs)
        print(f"Epoch {epoch:03d} | Loss: {total_loss/N:.4f}  CE: {total_ce/N:.4f}  Hyp: {total_hyp/N:.4f}")

    save_path = f"adapter_epoch_{target_epoch:03d}.pt"
    torch.save({
        "epoch":           target_epoch,
        "adapter_weights": [p.data.cpu() for p in trainable_params],
        "optimizer_state": optimizer.state_dict(),
    }, save_path)
    print(f"\nCheckpoint saved → {save_path}")

# =============================================================================
# SECTION 6 — Evaluation
# =============================================================================
try:
    print("\n--- EVALUATION ---")
    TASK_CONFIGS = {
        "boolean_expressions": ["True", "False"],
        "date_understanding": ["A", "B", "C", "D", "E", "F"],
        "logical_deduction_five_objects": ["A", "B", "C", "D", "E"],
        "navigate": ["Yes", "No"],
        "ruin_names": ["A", "B", "C", "D"],
        "snarks": ["A", "B"],
    }

    OPTION_TOKENS = {}
    for task_name, options in TASK_CONFIGS.items():
        for opt in options:
            if opt not in OPTION_TOKENS:
                tok_ids = tokenizer.encode(" " + opt, add_special_tokens=False)
                OPTION_TOKENS[opt] = tok_ids[0]

    def predict_mc(model, prompt, valid_options):
        msgs = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Answer:"
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :].float()
            probs = F.softmax(logits, dim=-1)
        raw = [probs[0, OPTION_TOKENS[opt]].item() for opt in valid_options]
        total = sum(raw)
        if total == 0: return valid_options[0], 0.0
        renorm = [r / total for r in raw]
        best = renorm.index(max(renorm))
        return valid_options[best], renorm[best]

    try:
        with open("test_set_ids.json", "r") as f:
            test_ids = json.load(f)
        TEST_TASKS = [r for r in golden_records if r.get("id") in test_ids]
        if len(TEST_TASKS) == 0:
            raise ValueError("test_set_ids.json found but no matching IDs in dataset.")
    except:
        # Fall back to the last 10 records
        TEST_TASKS = golden_records[-10:] if len(golden_records) >= 10 else golden_records

    if TEST_TASKS:
        print(f"Evaluating on {len(TEST_TASKS)} test records...\n")
        student_model.eval()
        correct = 0

        print(f"{'ID':<6} {'Task':<30} {'Ans':>5} | {'Pred':>5}")
        print("-" * 65)
        for t in TEST_TASKS:
            valid_options = TASK_CONFIGS[t["task"]]
            pred, prob = predict_mc(student_model, t["student_prompt"], valid_options)
            is_correct = (pred == t["target"])
            correct += is_correct
            mark = "✅" if is_correct else "❌"
            print(f"{t.get('id', '?'):<6} {t['task'][:28]:<30} {t['target']:>5} | {pred:>5} {mark}")

        print("-" * 65)
        print(f"Accuracy: {correct}/{len(TEST_TASKS)} ({correct/len(TEST_TASKS):.0%})")
    else:
        print("No test records found for evaluation.")
except Exception as e:
    print(f"Evaluation failed: {e}")

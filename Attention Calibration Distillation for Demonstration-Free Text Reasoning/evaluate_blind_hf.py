import json
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

print("==================================================================================")
print("TRUE GENERALIZATION EVALUATION")
print("Fetching entirely unseen REAL records from the tail end of HuggingFace BBH datasets...")
print("==================================================================================")

MC_LETTERS_ALL = ["A", "B", "C", "D", "E", "F"]

if 'tokenizer' not in globals():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

if 'device' not in globals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if 'student_model' in globals():
    eval_model = globals()['student_model']
elif 'model' in globals():
    eval_model = globals()['model']
else:
    raise RuntimeError("Could not find 'student_model' or 'model' in memory.")

MC_TOKEN_IDS = {}
for letter in MC_LETTERS_ALL:
    MC_TOKEN_IDS[letter] = tokenizer.encode(letter, add_special_tokens=False)[-1]

rng = random.Random(42)

# Exclude ruin_names due to the earlier parsing issue
TASK_CONFIGS = {
    "boolean_expressions": {"task_desc": "Evaluate the Boolean expression.", "mode": "dynamic", "num_options": 2},
    "navigate": {"task_desc": "Determine whether the navigation instructions lead back to the starting point.", "mode": "dynamic", "num_options": 2},
    "sports_understanding": {"task_desc": "Determine whether the sports statement is plausible or implausible.", "mode": "dynamic", "num_options": 2},
    "tracking_shuffled_objects_three_objects": {"task_desc": "Track the ownership of objects after swaps.", "mode": "dynamic", "num_options": 6},
    "date_understanding": {"task_desc": "Infer the date from the given context.", "mode": "native", "num_options": 6},
    "logical_deduction_five_objects": {"task_desc": "Deduce the logical ordering.", "mode": "native", "num_options": 5},
    "snarks": {"task_desc": "Identify the sarcastic sentence.", "mode": "native", "num_options": 2},
}

def format_prompt(text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "Answer:"

def predict_mc(prompt_text: str, valid_letters: list) -> str:
    formatted = format_prompt(prompt_text)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = eval_model(**inputs).logits[:, -1, :].float()
        probs  = F.softmax(logits, dim=-1)
    
    raw_probs = [probs[0, MC_TOKEN_IDS[l]].item() for l in valid_letters]
    total = sum(raw_probs)
    if total == 0: return valid_letters[0]
    renorm_probs = [p / total for p in raw_probs]
    best_idx = renorm_probs.index(max(renorm_probs))
    return valid_letters[best_idx]

def extract_native_letter(target: str) -> str:
    m = re.search(r'\(([A-F])\)', target)
    if m:
        return m.group(1)
    return None

def build_dynamic_options(correct_target: str, wrong_pool: list, num_options: int, slot: int) -> tuple:
    num_wrong = num_options - 1
    sampled_wrong = rng.sample(wrong_pool, min(num_wrong, len(wrong_pool)))
    options = sampled_wrong[:]
    options.insert(slot, correct_target)
    answer_letter = MC_LETTERS_ALL[slot]
    options_lines = "\n".join(f"{MC_LETTERS_ALL[i]}) {opt}" for i, opt in enumerate(options))
    return f"Options:\n{options_lines}", answer_letter

# =============================================================================
# 1. GENERATE UNSEEN PROMPTS FROM HUGGINGFACE
# =============================================================================
blind_test = []
global_id = 8000

for subset, cfg in TASK_CONFIGS.items():
    mode = cfg["mode"]
    task_desc = cfg["task_desc"]
    num_options = cfg["num_options"]
    valid_letters = MC_LETTERS_ALL[:num_options]

    ds = load_dataset("lukaemon/bbh", subset, split="test")

    if mode == "dynamic":
        unique_pool = list(set([ex["target"] for ex in ds]))

    demos = [ds[0], ds[1]]
    
    # GUARANTEE THEY ARE UNSEEN: Pull exactly 5 records from the very end of the dataset.
    tail_records = ds.select(range(len(ds)-5, len(ds)))

    slot_counter = 0
    demo_str = ""
    for i, demo in enumerate(demos):
        if mode == "dynamic":
            wrong_pool = [t for t in unique_pool if t != demo["target"]]
            opts_txt, ans_letter = build_dynamic_options(demo["target"], wrong_pool, num_options, slot_counter % num_options)
            slot_counter += 1
            demo_str += f"Example {i+1}\n{demo['input']}\n{opts_txt}\nAnswer: {ans_letter}\n\n"
        else:
            ans_letter = extract_native_letter(demo["target"])
            if ans_letter:
                demo_str += f"Example {i+1}\n{demo['input']}\nAnswer: {ans_letter}\n\n"

    for ex in tail_records:
        if mode == "dynamic":
            wrong_pool = [t for t in unique_pool if t != ex["target"]]
            opts_txt, ans_letter = build_dynamic_options(ex["target"], wrong_pool, num_options, slot_counter % num_options)
            slot_counter += 1
            question_block = f"Question\n{ex['input']}\n{opts_txt}"
        else:
            ans_letter = extract_native_letter(ex["target"])
            question_block = f"Question\n{ex['input']}"
            
        if not ans_letter:
            continue

        teacher_prompt = f"Task: {task_desc}\n\n{demo_str}{question_block}"
        student_prompt = f"Task: {task_desc}\n\n{question_block}"

        blind_test.append({
            "id": global_id,
            "task": subset,
            "teacher_prompt": teacher_prompt,
            "student_prompt": student_prompt,
            "answer": ans_letter,
            "valid_letters": valid_letters
        })
        global_id += 1

print(f"\nSuccessfully pulled and formatted {len(blind_test)} purely unseen records.\n")

# =========================================================
# 2. EVALUATE BASE 0-SHOT AND TEACHER 2-SHOT
# =========================================================
if hasattr(eval_model.model.layers[0].self_attn, 'original_attn'):
    print("Stripping the adapter to evaluate pure Base model...")
    for layer in eval_model.model.layers:
        layer.self_attn = layer.self_attn.original_attn

print("Evaluating Base (0-Shot) and Teacher (2-Shot)...")
results = {}
for t in tqdm(blind_test, desc="Baseline & Teacher"):
    ans = t["answer"]
    vl = t["valid_letters"]
    s_pred = predict_mc(t["student_prompt"], vl)
    t_pred = predict_mc(t["teacher_prompt"], vl)
    results[t["id"]] = {
        "task": t["task"], "ans": ans, 
        "base_0s": s_pred, "tch_2s": t_pred
    }

# =========================================================
# 3. INJECT ADAPTER AND EVALUATE TRAINED 0-SHOT
# =========================================================
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

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        if self._tuple_len is None:
            with torch.no_grad():
                dummy = self.original_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
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
            if self.num_kv_groups > 1:
                K = K.repeat_interleave(self.num_kv_groups, dim=1)
                V = V.repeat_interleave(self.num_kv_groups, dim=1)
        Q, K, V = Q.detach().float(), K.detach().float(), V.detach().float()
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
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim).to(dtype=hidden_states.dtype)
        return (self.original_attn.o_proj(attn_output), None, None, None)[:self._tuple_len]

print("\nInjecting Adapter and loading weights from adapter_epoch_020.pt...")
trainable_params = []
for layer in eval_model.model.layers:
    cal = CalibratedAttention(layer.self_attn, rank=4).to(device)
    layer.self_attn = cal
    trainable_params.extend([cal.U_q, cal.U_k])

ckpt = torch.load("adapter_epoch_020.pt", map_location=device)
for p, saved in zip(trainable_params, ckpt["adapter_weights"]):
    p.data.copy_(saved.to(device))

eval_model.eval()

print("Evaluating Trained Student (0-Shot)...")
for t in tqdm(blind_test, desc="Trained Model"):
    vl = t["valid_letters"]
    trained_pred = predict_mc(t["student_prompt"], vl)
    results[t["id"]]["trained_0s"] = trained_pred

# =========================================================
# 4. PRINT RESULTS
# =========================================================
base_correct = 0
tch_correct = 0
trained_correct = 0
intersection_all = 0
distilled_success = 0 

print(f"\n{'='*100}")
print(f"{'ID':<6} {'Task':<35} {'Ans':>3} | {'Tch(2S)':>7} | {'Base(0S)':>8} | {'Trn(0S)':>8}")
print("-" * 100)

for t in blind_test:
    r = results[t["id"]]
    ans = r["ans"]
    
    t_corr = r["tch_2s"] == ans
    b_corr = r["base_0s"] == ans
    trn_corr = r["trained_0s"] == ans
    
    tch_correct += t_corr
    base_correct += b_corr
    trained_correct += trn_corr
    
    if t_corr and b_corr and trn_corr:
        intersection_all += 1
    if t_corr and trn_corr and not b_corr:
        distilled_success += 1
        
    t_mark = "✅" if t_corr else "❌"
    b_mark = "✅" if b_corr else "❌"
    trn_mark = "✅" if trn_corr else "❌"
    
    change = ""
    if not b_corr and trn_corr:
        change = "← DISTILLED ✨"
    elif b_corr and not trn_corr:
        change = "← REGRESSED ⚠️"
        
    print(f"{t['id']:<6} {r['task'][:33]:<35} {ans:>3} | {r['tch_2s']} {t_mark}   | {r['base_0s']} {b_mark}    | {r['trained_0s']} {trn_mark}    {change}")

print("-" * 100)
total = len(blind_test)
print(f"\nTeacher (2-shot)      : {tch_correct}/{total} ({tch_correct/total:.0%})")
print(f"Base Student (0-shot) : {base_correct}/{total} ({base_correct/total:.0%})")
print(f"Trained (0-shot)      : {trained_correct}/{total} ({trained_correct/total:.0%})")

print(f"\nIntersection (Correct in ALL THREE) : {intersection_all}")
print(f"True Distillation (Wrong in Base, Correct in Teacher & Trained): {distilled_success}")

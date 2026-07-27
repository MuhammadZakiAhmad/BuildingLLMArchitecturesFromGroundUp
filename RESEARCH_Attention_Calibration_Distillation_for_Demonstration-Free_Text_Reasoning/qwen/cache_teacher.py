"""
cache_teacher.py
Pre-computes and caches the teacher hidden states to disk for Qwen2.5-3B.
Run this first, then RESTART YOUR KERNEL to wipe VRAM before running train_adapter_v2.py.
"""
import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_NAME   = "Qwen/Qwen2.5-3B-Instruct"
DATA_FILE    = "bbh_golden_v2.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading tokenizer and teacher model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
teacher_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="eager"
).to(device)
teacher_model.eval()

for p in teacher_model.parameters():
    p.requires_grad = False

print(f"Loading {DATA_FILE}...")
with open(DATA_FILE, "r") as f:
    golden_records = json.load(f)

random.seed(42)
_shuffled = golden_records.copy()
random.shuffle(_shuffled)

# DYNAMIC 80/20 SPLIT
train_ratio_idx = int(len(_shuffled) * 0.8)
TRAIN_TASKS = _shuffled[:train_ratio_idx]
print(f"Dataset has {len(_shuffled)} records. Caching teacher states for {len(TRAIN_TASKS)} training records.")

def _format(text: str) -> str:
    msgs = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Answer:"

MC_LETTERS_ALL = ["A", "B", "C", "D", "E", "F"]
MC_TOKEN_IDS = {
    l: tokenizer.encode(l, add_special_tokens=False)[-1]
    for l in MC_LETTERS_ALL
}

print("Pre-computing teacher hidden states...")
train_pairs = []
for t in tqdm(TRAIN_TASKS, desc="Teacher targets"):
    t_inputs = tokenizer(_format(t["teacher_prompt"]), return_tensors="pt").to(device)
    s_inputs = tokenizer(_format(t["student_prompt"]), return_tensors="pt").to(device)
    
    with torch.no_grad():
        out_t = teacher_model(**t_inputs, output_hidden_states=True)
        # Move immediately to CPU so we don't blow up VRAM
        t_hs  = [
            out_t.hidden_states[l][:, -1, :].float().cpu()
            for l in range(len(out_t.hidden_states))
        ]
        
    correct_idx = torch.tensor(
        [t["valid_letters"].index(t["answer"])], dtype=torch.long
    )
    valid_ids = [MC_TOKEN_IDS[l] for l in t["valid_letters"]]
    
    # Store s_inputs to CPU as well
    s_inputs_cpu = {k: v.cpu() for k, v in s_inputs.items()}
    train_pairs.append((s_inputs_cpu, t_hs, correct_idx, valid_ids, t["id"]))

torch.save(train_pairs, "teacher_hidden_states.pt")
print(f"✅ Saved {len(train_pairs)} pairs to teacher_hidden_states.pt")
print("🔥 CRITICAL: Restart your Colab Kernel / Runtime NOW to completely free VRAM before training!")

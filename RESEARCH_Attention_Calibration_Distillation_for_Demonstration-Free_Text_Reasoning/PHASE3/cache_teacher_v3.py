"""
cache_teacher_v3.py
Phase 3 Caching Script for Hyper-ICL Sequence Distillation

Fixes:
1. Slices the exact sequence of query tokens to match the student prompt.
2. Removes " Let's think step by step." from teacher prompt to ensure token alignment.
3. Supports final_golden_records.json format (uses 'target' key).
4. Ultra-sparse caching: Only saves TARGET_LAYERS to disk to prevent RAM/Disk bloat.
5. Strict 80 Train / 10 Test split.
"""
import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME    = "Qwen/Qwen2.5-3B-Instruct"
DATA_FILE     = "pure_golden_records.json"
OUTPUT_FILE   = "teacher_hidden_states_phase3.pt"
TARGET_LAYERS = [-4, -3, -2, -1] # Must match train_adapter_v3.py
NUM_TRAIN     = 42
NUM_TEST      = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Checking for existing models in memory...")
if 'tokenizer' not in globals():
    print("Loading tokenizer natively...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
else:
    print("Tokenizer found in memory!")

if 'teacher_model' not in globals():
    print("Loading teacher model natively...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
else:
    print("Teacher model found in memory!")

# =============================================================================
# DATASET PARSING
# =============================================================================
print(f"Loading {DATA_FILE}...")
with open(DATA_FILE, "r") as f:
    golden_records = json.load(f)

# Optional: Ensure we have enough records
assert len(golden_records) >= (NUM_TRAIN + NUM_TEST), f"Not enough records. Found {len(golden_records)}."

random.seed(42)
_shuffled = golden_records.copy()
random.shuffle(_shuffled)

TRAIN_TASKS = _shuffled[:NUM_TRAIN]
TEST_TASKS  = _shuffled[NUM_TRAIN:NUM_TRAIN+NUM_TEST]

print(f"Total golden records: {len(golden_records)}")
print(f"Allocated {len(TRAIN_TASKS)} for training and {len(TEST_TASKS)} for testing.")

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

# =============================================================================
# CACHING LOOP
# =============================================================================
def find_lcs(a, b):
    for i in range(1, min(len(a), len(b)) + 1):
        if a[-i] != b[-i]:
            return i - 1
    return min(len(a), len(b))

# To map negative indices (e.g., -1) to actual layers
total_layers = len(teacher_model.model.layers)
actual_target_layers = [l if l >= 0 else total_layers + l for l in TARGET_LAYERS]

train_pairs = []
print("Pre-computing teacher hidden states for TRAIN set...")

for t in tqdm(TRAIN_TASKS, desc="Caching Teacher States"):
    # 1. Clean the teacher prompt so it aligns perfectly with the student prompt's end.
    # The JSON's teacher_prompt ends with "A: Let's think step by step."
    # The student_prompt ends with "A:"
    # Because of causal attention, the teacher's hidden states for the query 
    # are identical whether "Let's think" is appended or not.
    # We strip it to ensure perfectly matching token sequence lengths for slicing.
    raw_teacher = t["teacher_prompt"]
    if raw_teacher.endswith(" Let's think step by step."):
        raw_teacher = raw_teacher[:-len(" Let's think step by step.")]
    
    # We use chat template so the model sees it as a proper user instruction
    # But wait, in the previous script `_format` was used. We should format both identically.
    def _format(text: str) -> str:
        msgs = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    
    # Format both
    formatted_t = _format(raw_teacher)
    formatted_s = _format(t["student_prompt"])
    
    t_inputs = tokenizer(formatted_t, return_tensors="pt").to(device)
    s_inputs = tokenizer(formatted_s, return_tensors="pt").to(device)
    
    # [CRITICAL FIX]: Dynamic LCS Slicing
    # Chat templates cause the prefix (e.g., "\nQ:" vs "user\nQ:") to tokenize differently.
    # We must find the exact matching suffix of token IDs to guarantee perfect alignment.
    t_ids = t_inputs["input_ids"][0].tolist()
    s_ids = s_inputs["input_ids"][0].tolist()
    lcs_len = find_lcs(s_ids, t_ids)
    
    if lcs_len < 5:
        print(f"Warning: LCS is dangerously short ({lcs_len} tokens) for record {t['record_id']}")
    
    with torch.no_grad():
        out_t = teacher_model(**t_inputs, output_hidden_states=True)
        
        # [Fix 1, 2, 4]: Ultra-Sparse + Sequence Slicing
        # out_t.hidden_states[0] is embedding layer. Transformer layers start at 1.
        t_hs_sparse = {}
        for layer_idx in actual_target_layers:
            # Get the full hidden states for this transformer layer
            full_hs = out_t.hidden_states[layer_idx + 1]
            # Slice only the matching LCS tokens
            sliced_hs = full_hs[:, -lcs_len:, :].float().cpu()
            t_hs_sparse[layer_idx] = sliced_hs
            
    # [Fix 3]: Safe dataset key access (using 'target')
    valid_options = TASK_CONFIGS[t["task"]]
    target_letter = t["target"]
    if target_letter in valid_options:
        correct_idx = valid_options.index(target_letter)
    else:
        # Fallback if target is somehow corrupted
        correct_idx = 0 
        
    valid_ids = [OPTION_TOKENS[opt] for opt in valid_options]
    
    # Store s_inputs to CPU
    s_inputs_cpu = {k: v.cpu() for k, v in s_inputs.items()}
    
    train_pairs.append((s_inputs_cpu, t_hs_sparse, correct_idx, valid_ids))

torch.save(train_pairs, OUTPUT_FILE)
print(f"✅ Saved {len(train_pairs)} pairs to {OUTPUT_FILE}")

# Save the test set IDs to a separate file so we can evaluate on them later
test_ids = [t.get("id") for t in TEST_TASKS]
with open("test_set_ids.json", "w") as f:
    json.dump(test_ids, f, indent=2)

print("✅ Saved test_set_ids.json")
print("🔥 CRITICAL: Restart your Colab Kernel / Runtime NOW to completely free VRAM before running train_adapter_v3.py!")

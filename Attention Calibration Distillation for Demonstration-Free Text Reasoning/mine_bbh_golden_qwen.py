"""
mine_bbh_golden_qwen.py

Run inside your Colab notebook with:
    %run -i mine_bbh_golden_qwen.py

This script will automatically load the Qwen model and tokenizer natively if they aren't already loaded.
"""

import json
import re
import random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

TEACHER_CONFIDENCE_THRESHOLD = 0.60

MC_LETTERS_ALL = ["A", "B", "C", "D", "E", "F"]

# =============================================================================
# AUTO-LOAD MODEL & TOKENIZER
# =============================================================================
if 'tokenizer' not in globals():
    from transformers import AutoTokenizer
    print("Loading tokenizer natively...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

if 'device' not in globals():
    print("Setting device natively...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

if 'model' not in globals():
    from transformers import AutoModelForCausalLM
    print("Loading model natively...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct", 
        torch_dtype=torch.bfloat16, 
        attn_implementation="eager"
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

# =============================================================================
# SETUP
# =============================================================================
# Robust token extraction for Qwen
MC_TOKEN_IDS = {}
for letter in MC_LETTERS_ALL:
    # Get the token ID for the letter. Taking the last token in case of prefixes.
    MC_TOKEN_IDS[letter] = tokenizer.encode(letter, add_special_tokens=False)[-1]

rng = random.Random(42)

TASK_CONFIGS = {
    "boolean_expressions": {"task_desc": "Evaluate the Boolean expression.", "mode": "dynamic", "num_options": 2},
    "navigate": {"task_desc": "Determine whether the navigation instructions lead back to the starting point.", "mode": "dynamic", "num_options": 2},
    "sports_understanding": {"task_desc": "Determine whether the sports statement is plausible or implausible.", "mode": "dynamic", "num_options": 2},
    "tracking_shuffled_objects_three_objects": {"task_desc": "Track the ownership of objects after swaps.", "mode": "dynamic", "num_options": 6},
    "date_understanding": {"task_desc": "Infer the date from the given context.", "mode": "native", "num_options": 6},
    "logical_deduction_five_objects": {"task_desc": "Deduce the logical ordering.", "mode": "native", "num_options": 5},
    "snarks": {"task_desc": "Identify the sarcastic sentence.", "mode": "native", "num_options": 2},
    "ruin_names": {"task_desc": "Identify the humorous edit to the artist or movie name.", "mode": "native", "num_options": 4},
}

def format_prompt(text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "Answer:"

def predict_mc(prompt_text: str, valid_letters: list) -> tuple:
    formatted = format_prompt(prompt_text)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :].float()
        probs  = F.softmax(logits, dim=-1)
    
    # Extract only valid letters
    raw_probs = [probs[0, MC_TOKEN_IDS[l]].item() for l in valid_letters]
    
    # Re-normalize over the valid choices to isolate confidence
    total = sum(raw_probs)
    if total == 0:
        return valid_letters[0], 0.0
    
    renorm_probs = [p / total for p in raw_probs]
    best_idx = renorm_probs.index(max(renorm_probs))
    
    return valid_letters[best_idx], renorm_probs[best_idx]

def extract_native_letter(target: str) -> str:
    m = re.search(r'\(([A-F])\)', target)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract native letter from: '{target}'")

def build_dynamic_options(correct_target: str, wrong_pool: list, num_options: int, slot: int) -> tuple:
    num_wrong = num_options - 1
    sampled_wrong = rng.sample(wrong_pool, min(num_wrong, len(wrong_pool)))
    options = sampled_wrong[:]
    options.insert(slot, correct_target)
    answer_letter = MC_LETTERS_ALL[slot]
    options_lines = "\n".join(f"{MC_LETTERS_ALL[i]}) {opt}" for i, opt in enumerate(options))
    return f"Options:\n{options_lines}", answer_letter

# =============================================================================
# MINING
# =============================================================================
golden_records = []
global_id = 1

for subset, cfg in TASK_CONFIGS.items():
    mode = cfg["mode"]
    task_desc = cfg["task_desc"]
    num_options = cfg["num_options"]
    valid_letters = MC_LETTERS_ALL[:num_options]

    print(f"\nTask: {subset} | Mode: {mode} | Options: A-{valid_letters[-1]}")
    ds = load_dataset("lukaemon/bbh", subset, split="test")

    if mode == "dynamic":
        unique_pool = list(set([ex["target"] for ex in ds]))

    demos = [ds[0], ds[1]]
    test_records = ds.select(range(2, len(ds)))

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
            demo_str += f"Example {i+1}\n{demo['input']}\nAnswer: {ans_letter}\n\n"

    for ex in tqdm(test_records, desc=subset):
        if mode == "dynamic":
            wrong_pool = [t for t in unique_pool if t != ex["target"]]
            opts_txt, ans_letter = build_dynamic_options(ex["target"], wrong_pool, num_options, slot_counter % num_options)
            slot_counter += 1
            question_block = f"Question\n{ex['input']}\n{opts_txt}"
        else:
            ans_letter = extract_native_letter(ex["target"])
            question_block = f"Question\n{ex['input']}"

        teacher_prompt = f"Task: {task_desc}\n\n{demo_str}{question_block}"
        student_prompt = f"Task: {task_desc}\n\n{question_block}"

        teacher_pred, teacher_conf = predict_mc(teacher_prompt, valid_letters)
        student_pred, student_conf = predict_mc(student_prompt, valid_letters)

        teacher_correct = (teacher_pred == ans_letter)
        student_correct = (student_pred == ans_letter)

        if teacher_correct and teacher_conf >= TEACHER_CONFIDENCE_THRESHOLD and not student_correct:
            golden_records.append({
                "id": global_id,
                "task": subset,
                "teacher_prompt": teacher_prompt,
                "student_prompt": student_prompt,
                "answer": ans_letter,
                "teacher_pred": teacher_pred,
                "teacher_conf": round(teacher_conf, 4),
                "student_pred": student_pred,
                "student_conf": round(student_conf, 4),
                "valid_letters": valid_letters,
            })
        global_id += 1

print("\n\n" + "="*60)
print(f"MINING COMPLETE. Found {len(golden_records)} total raw golden records.")
print("Performing strict mathematical distribution balancing...")

# =============================================================================
# STRICT DISTRIBUTION TRUNCATION
# =============================================================================
binary_records = defaultdict(list)
multi_records = defaultdict(list)

for r in golden_records:
    ans = r["answer"]
    if len(r["valid_letters"]) == 2:
        binary_records[ans].append(r)
    else:
        multi_records[ans].append(r)

final_dataset = []

if binary_records:
    binary_keys = ["A", "B"]
    min_binary = min([len(binary_records.get(k, [])) for k in binary_keys])
    print(f"Binary Tasks: Smallest class has {min_binary} records. Truncating A and B to {min_binary}.")
    for k in binary_keys:
        if k in binary_records:
            final_dataset.extend(rng.sample(binary_records[k], min_binary))

if multi_records:
    active_keys = [k for k in MC_LETTERS_ALL if k in multi_records]
    min_multi = min([len(multi_records[k]) for k in active_keys])
    print(f"Multi-Choice Tasks: Smallest class has {min_multi} records. Truncating {', '.join(active_keys)} to {min_multi}.")
    for k in active_keys:
        final_dataset.extend(rng.sample(multi_records[k], min_multi))

rng.shuffle(final_dataset)

with open("bbh_golden_v2.json", "w") as f:
    json.dump(final_dataset, f, indent=2)

print("\nFinal Balanced Dataset saved to 'bbh_golden_v2.json'.")
print(f"Total Size: {len(final_dataset)} records.")

from collections import Counter
print("\nFinal Binary Distribution:")
print(Counter([r['answer'] for r in final_dataset if len(r['valid_letters'])==2]))

print("\nFinal Multi-Choice Distribution:")
print(Counter([r['answer'] for r in final_dataset if len(r['valid_letters'])>2]))

"""
mine_bbh_golden_v2.py

Run inside your Colab notebook with:
    %run -i mine_bbh_golden_v2.py

Requires the following already loaded in the notebook namespace:
    - model       : SmolLM2-1.7B-Instruct (bfloat16, on device)
    - tokenizer   : matching AutoTokenizer
    - device      : torch.device

Outputs:
    bbh_golden_v2.json      — teacher correct+confident, student wrong
    bbh_easy_v2.json        — both correct
    bbh_hard_v2.json        — both wrong
    bbh_anomalous_v2.json   — teacher wrong, student correct
"""

import json
import re
import random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

TEACHER_CONFIDENCE_THRESHOLD = 0.60

# All supported MC letters
MC_LETTERS_ALL = ["A", "B", "C", "D", "E", "F"]

# Token IDs resolved against the loaded tokenizer
MC_TOKEN_IDS = {
    letter: tokenizer.encode(letter, add_special_tokens=False)[0]
    for letter in MC_LETTERS_ALL
}

# Seeded RNG for reproducible wrong-answer sampling
rng = random.Random(42)

# ── Task definitions ──────────────────────────────────────────────────────────
#
# mode="dynamic" : options are built from the dataset target pool.
#                  num_options drives the A-B / A-F split.
#                  Round-robin label assignment applies.
#
# mode="native"  : options are already embedded in the BBH input text.
#                  We pass the raw input directly; no options are appended.
#                  Round-robin cannot apply (options are baked into text).

TASK_CONFIGS = {
    "boolean_expressions": {
        "task_desc": "Evaluate the Boolean expression.",
        "mode": "dynamic",
        "num_options": 2,       # True / False  →  A-B
    },
    "navigate": {
        "task_desc": "Determine whether the navigation instructions lead back to the starting point.",
        "mode": "dynamic",
        "num_options": 2,       # Yes / No  →  A-B
    },
    "sports_understanding": {
        "task_desc": "Determine whether the sports statement is plausible or implausible.",
        "mode": "dynamic",
        "num_options": 2,       # yes / no  →  A-B
    },
    "tracking_shuffled_objects_three_objects": {
        "task_desc": "Track the ownership of objects after swaps.",
        "mode": "dynamic",
        "num_options": 6,       # person names  →  A-F
    },
    "date_understanding": {
        "task_desc": "Infer the date from the given context.",
        "mode": "native",
        "num_options": 6,       # native (A)-(F) in text
    },
    "logical_deduction_five_objects": {
        "task_desc": "Deduce the logical ordering.",
        "mode": "native",
        "num_options": 5,       # native (A)-(E) in text
    },
    "snarks": {
        "task_desc": "Identify the sarcastic sentence.",
        "mode": "native",
        "num_options": 2,       # native (A)-(B) in text
    },
    "ruin_names": {
        "task_desc": "Identify the humorous edit to the artist or movie name.",
        "mode": "native",
        "num_options": 4,       # native (A)-(D) in text
    },
}

# ── Core helpers ──────────────────────────────────────────────────────────────

def format_prompt(text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + "Answer:"
    )


def predict_mc(prompt_text: str, valid_letters: list) -> tuple:
    """
    Returns (predicted_letter, confidence) restricted to valid_letters only.
    Confidence is the raw softmax probability assigned to the winning letter.
    """
    formatted = format_prompt(prompt_text)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :].float()
        probs  = F.softmax(logits, dim=-1)
    mc_probs = [probs[0, MC_TOKEN_IDS[l]].item() for l in valid_letters]
    best_idx = mc_probs.index(max(mc_probs))
    return valid_letters[best_idx], mc_probs[best_idx]


def extract_native_letter(target: str) -> str:
    """
    Extracts the answer letter from a native BBH target string.
    Handles formats like '(B)', '(B) 12/25/1937', '(A) The quail is rightmost'.
    """
    m = re.search(r'\(([A-F])\)', target)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract native letter from: '{target}'")


def build_dynamic_options(
    correct_target: str,
    wrong_pool: list,
    num_options: int,
    slot: int
) -> tuple:
    """
    Places `correct_target` at position `slot` among `num_options` options.
    Fills remaining slots with samples from `wrong_pool`.
    Returns (options_text_str, answer_letter).
    """
    num_wrong = num_options - 1
    sampled_wrong = rng.sample(wrong_pool, min(num_wrong, len(wrong_pool)))

    # Build the ordered option list
    options = sampled_wrong[:]           # num_options - 1 items
    options.insert(slot, correct_target) # inject correct at designated slot

    answer_letter  = MC_LETTERS_ALL[slot]
    options_lines  = "\n".join(
        f"{MC_LETTERS_ALL[i]}) {opt}" for i, opt in enumerate(options)
    )
    options_text   = f"Options:\n{options_lines}"

    return options_text, answer_letter

# ── Main mining loop ──────────────────────────────────────────────────────────

buckets = {
    "golden":    [],   # teacher correct + confident, student wrong
    "easy":      [],   # both correct
    "hard":      [],   # both wrong
    "anomalous": [],   # teacher wrong, student correct
}
global_id = 1

for subset, cfg in TASK_CONFIGS.items():
    mode        = cfg["mode"]
    task_desc   = cfg["task_desc"]
    num_options = cfg["num_options"]
    valid_letters = MC_LETTERS_ALL[:num_options]

    print(f"\n{'='*64}")
    print(f"Task : {subset}")
    print(f"Mode : {mode}  |  Options : A-{valid_letters[-1]}")
    print(f"{'='*64}")

    ds = load_dataset("lukaemon/bbh", subset, split="test")

    # Build wrong-answer pool for dynamic tasks
    if mode == "dynamic":
        all_targets  = [ex["target"] for ex in ds]
        unique_pool  = list(set(all_targets))

    demos        = [ds[0], ds[1]]
    test_records = ds.select(range(2, len(ds)))

    # Build 2-shot demo block
    # slot_counter starts at 0 and advances only for dynamic tasks
    slot_counter = 0
    demo_str     = ""

    for i, demo in enumerate(demos):
        if mode == "dynamic":
            wrong_pool  = [t for t in unique_pool if t != demo["target"]]
            opts_txt, ans_letter = build_dynamic_options(
                demo["target"], wrong_pool, num_options, slot_counter % num_options
            )
            slot_counter += 1
            demo_str += f"Example {i+1}\n{demo['input']}\n{opts_txt}\nAnswer: {ans_letter}\n\n"
        else:  # native
            ans_letter  = extract_native_letter(demo["target"])
            # Raw input already contains the embedded options — append Answer only
            demo_str += f"Example {i+1}\n{demo['input']}\nAnswer: {ans_letter}\n\n"

    # Process test examples
    for ex in tqdm(test_records, desc=subset):
        if mode == "dynamic":
            wrong_pool  = [t for t in unique_pool if t != ex["target"]]
            opts_txt, ans_letter = build_dynamic_options(
                ex["target"], wrong_pool, num_options, slot_counter % num_options
            )
            slot_counter  += 1
            question_block = f"Question\n{ex['input']}\n{opts_txt}"
        else:  # native
            ans_letter     = extract_native_letter(ex["target"])
            # Options already in the text — no extra block needed
            question_block = f"Question\n{ex['input']}"

        task_header    = f"Task: {task_desc}\n\n"
        teacher_prompt = task_header + demo_str + question_block
        student_prompt = task_header + question_block

        teacher_pred, teacher_conf = predict_mc(teacher_prompt, valid_letters)
        student_pred, student_conf = predict_mc(student_prompt, valid_letters)

        teacher_correct = (teacher_pred == ans_letter)
        student_correct = (student_pred == ans_letter)

        record = {
            "id":            global_id,
            "task":          subset,
            "teacher_prompt": teacher_prompt,
            "student_prompt": student_prompt,
            "answer":        ans_letter,
            "teacher_pred":  teacher_pred,
            "teacher_conf":  round(teacher_conf, 4),
            "student_pred":  student_pred,
            "student_conf":  round(student_conf, 4),
            "valid_letters": valid_letters,
        }
        global_id += 1

        # Route to the correct bucket
        if teacher_correct and teacher_conf >= TEACHER_CONFIDENCE_THRESHOLD and not student_correct:
            buckets["golden"].append(record)
        elif teacher_correct and student_correct:
            buckets["easy"].append(record)
        elif not teacher_correct and not student_correct:
            buckets["hard"].append(record)
        else:
            buckets["anomalous"].append(record)

# ── Shuffle, save, and report ─────────────────────────────────────────────────

print("\n\n" + "="*64)
print("MINING COMPLETE — Shuffling and saving buckets")
print("="*64)

for bucket_name, records in buckets.items():
    rng.shuffle(records)
    fname = f"bbh_{bucket_name}_v2.json"
    with open(fname, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  {fname:<30} : {len(records):>5} records")

# Label distribution within golden bucket
golden = buckets["golden"]
if golden:
    print("\nGolden label distribution:")
    from collections import Counter
    label_counts = Counter(r["answer"] for r in golden)
    for letter in MC_LETTERS_ALL:
        if letter in label_counts:
            pct = 100 * label_counts[letter] / len(golden)
            print(f"  {letter} : {label_counts[letter]:>4} ({pct:.1f}%)")

print("\nDone.")

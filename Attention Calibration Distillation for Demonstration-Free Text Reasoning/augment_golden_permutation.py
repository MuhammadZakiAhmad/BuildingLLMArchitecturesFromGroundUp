"""
augment_golden_permutation.py

Performs option permutation augmentation on golden records where answer=A.
Takes 30 of those records, swaps the correct answer to B/C/D/E, and re-runs
inference to verify they are still genuinely golden after the swap.

Run with: %run -i augment_golden_permutation.py
Requires: model, tokenizer, device already loaded in the Colab namespace.
"""

import json
import re
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── MC setup (reuses whatever is already loaded) ──────────────────────────────
MC_LETTERS = ["A", "B", "C", "D", "E", "F"]
MC_TOKEN_IDS = {
    l: tokenizer.encode(l, add_special_tokens=False)[0]
    for l in MC_LETTERS
}

aug_rng = random.Random(99)  # separate seed from mining script

# ── Helpers ───────────────────────────────────────────────────────────────────

def format_prompt(text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + "Answer:"
    )


def predict_mc(prompt_text: str, valid_letters: list) -> tuple:
    """Returns (letter, raw_vocab_prob) for the winning letter among valid_letters."""
    formatted = format_prompt(prompt_text)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :].float()
        probs  = F.softmax(logits, dim=-1)
    mc_probs = [probs[0, MC_TOKEN_IDS[l]].item() for l in valid_letters]
    best_idx = mc_probs.index(max(mc_probs))
    return valid_letters[best_idx], mc_probs[best_idx]


def swap_options_in_block(block: str, from_letter: str, to_letter: str, valid_letters: list):
    """
    Swaps the text of two option positions in a question block.
    Handles 'A) text' (dynamic) and '(A) text' (native) formats.
    Returns (new_block, success).
    """
    lines = block.split('\n')
    option_map = {}  # letter -> (line_index, option_text, format)

    for i, line in enumerate(lines):
        m = re.match(r'^([A-F])\) (.+)$', line)           # dynamic: A) text
        if m and m.group(1) in valid_letters:
            option_map[m.group(1)] = (i, m.group(2), 'dynamic')
            continue
        m = re.match(r'^\(([A-F])\) (.+)$', line)         # native: (A) text
        if m and m.group(1) in valid_letters:
            option_map[m.group(1)] = (i, m.group(2), 'native')

    if from_letter not in option_map or to_letter not in option_map:
        return block, False

    idx_a, text_a, fmt = option_map[from_letter]
    idx_b, text_b, _   = option_map[to_letter]

    if fmt == 'dynamic':
        lines[idx_a] = f'{from_letter}) {text_b}'
        lines[idx_b] = f'{to_letter}) {text_a}'
    else:
        lines[idx_a] = f'({from_letter}) {text_b}'
        lines[idx_b] = f'({to_letter}) {text_a}'

    return '\n'.join(lines), True


def permute_record(record: dict, target_letter: str):
    """
    Swaps the correct answer from 'A' to target_letter in both prompts.
    Re-runs inference. Returns a new record dict if still golden, else None.
    """
    valid_letters = record["valid_letters"]

    if target_letter not in valid_letters:
        return None

    if "Question\n" not in record["student_prompt"]:
        return None

    # Isolate the test question block (appears once in student_prompt)
    prefix_s, q_block = record["student_prompt"].rsplit("Question\n", 1)

    # Swap inside the question block
    new_q_block, success = swap_options_in_block(
        q_block, "A", target_letter, valid_letters
    )
    if not success:
        return None

    new_student_prompt = prefix_s + "Question\n" + new_q_block

    # Apply the exact same text change to teacher_prompt (count=1 is safe
    # because "Question\n" appears exactly once in both prompts)
    new_teacher_prompt = record["teacher_prompt"].replace(
        "Question\n" + q_block,
        "Question\n" + new_q_block,
        1
    )

    # Re-run inference on swapped prompts
    teacher_pred, teacher_conf = predict_mc(new_teacher_prompt, valid_letters)
    student_pred, student_conf = predict_mc(new_student_prompt, valid_letters)

    teacher_correct = (teacher_pred == target_letter)
    student_correct = (student_pred == target_letter)

    if not teacher_correct or student_correct:
        return None  # No longer golden after swap — discard

    new_record = dict(record)
    new_record["teacher_prompt"] = new_teacher_prompt
    new_record["student_prompt"] = new_student_prompt
    new_record["answer"]         = target_letter
    new_record["teacher_pred"]   = teacher_pred
    new_record["teacher_conf"]   = round(teacher_conf, 4)
    new_record["student_pred"]   = student_pred
    new_record["student_conf"]   = round(student_conf, 4)
    new_record["augmented"]      = True
    new_record["original_answer"]= "A"

    return new_record

# ── Load and split golden records ─────────────────────────────────────────────

with open("bbh_golden_final.json", "r") as f:
    golden = json.load(f)

a_records     = [r for r in golden if r["answer"] == "A"]
non_a_records = [r for r in golden if r["answer"] != "A"]

print(f"Total golden records  : {len(golden)}")
print(f"Answer = A            : {len(a_records)}")
print(f"Answer != A           : {len(non_a_records)}")

# Pick 30 to permute, keep the rest as-is
aug_rng.shuffle(a_records)
to_permute   = a_records[:30]
keep_as_a    = a_records[30:]

print(f"\nPermuting 30 A-records — cycling targets through B, C, D, E ...")

# ── Permutation loop ──────────────────────────────────────────────────────────

TARGET_CYCLE = ["B", "C", "D", "E"]
new_records  = []
failed       = 0

for cycle_idx, rec in enumerate(tqdm(to_permute, desc="Permuting")):
    valid = rec["valid_letters"]

    # Restrict target cycle to letters that actually exist in this task
    valid_targets = [l for l in TARGET_CYCLE if l in valid]
    if not valid_targets:
        valid_targets = ["B"]  # binary task fallback

    target = valid_targets[cycle_idx % len(valid_targets)]
    result = permute_record(rec, target)

    if result is not None:
        new_records.append(result)
    else:
        # Teacher failed the swap — keep original as A
        keep_as_a.append(rec)
        failed += 1

print(f"\nPermutation complete:")
print(f"  Successfully golden after swap : {len(new_records)}")
print(f"  Teacher failed swap (kept as A): {failed}")

# ── Combine, shuffle, save ────────────────────────────────────────────────────

final_golden = non_a_records + keep_as_a + new_records
random.seed(42)
random.shuffle(final_golden)

with open("bbh_golden_final.json", "w") as f:
    json.dump(final_golden, f, indent=2)

print(f"\nFinal golden record count : {len(final_golden)}")
print("\nLabel distribution:")
from collections import Counter
label_counts = Counter(r["answer"] for r in final_golden)
for letter in MC_LETTERS:
    if letter in label_counts:
        pct = 100 * label_counts[letter] / len(final_golden)
        print(f"  {letter} : {label_counts[letter]:>3} ({pct:.1f}%)")

print("\nTask distribution:")
task_counts = Counter(r["task"] for r in final_golden)
for task, count in sorted(task_counts.items()):
    print(f"  {task:<45} : {count:>3}")

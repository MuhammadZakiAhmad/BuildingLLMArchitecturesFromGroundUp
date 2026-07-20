import json
import re
import copy
from collections import Counter

file_path = "bbh_golden_final_with_conf.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Ensure backup exists
with open("bbh_golden_final_with_conf.bak.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

def swap_options_in_block(block: str, from_letter: str, to_letter: str, valid_letters: list):
    lines = block.split('\n')
    option_map = {}

    for i, line in enumerate(lines):
        m = re.match(r'^([A-F])\) (.+)$', line)
        if m and m.group(1) in valid_letters:
            option_map[m.group(1)] = (i, m.group(2), 'dynamic')
            continue
        m = re.match(r'^\(([A-F])\) (.+)$', line)
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
    valid_letters = record["valid_letters"]
    original_answer = record["answer"]
    
    if target_letter not in valid_letters:
        return None
    
    if target_letter == original_answer:
        return record # No swap needed if it's already the target

    if "Question\n" not in record["student_prompt"]:
        return None

    prefix_s, q_block = record["student_prompt"].rsplit("Question\n", 1)
    
    new_q_block, success = swap_options_in_block(q_block, original_answer, target_letter, valid_letters)
    if not success:
        return None

    new_student_prompt = prefix_s + "Question\n" + new_q_block
    
    if "Question\n" + q_block not in record["teacher_prompt"]:
        return None
        
    new_teacher_prompt = record["teacher_prompt"].replace("Question\n" + q_block, "Question\n" + new_q_block, 1)

    new_record = copy.deepcopy(record)
    new_record["teacher_prompt"] = new_teacher_prompt
    new_record["student_prompt"] = new_student_prompt
    new_record["answer"] = target_letter
    
    if new_record.get("teacher_pred") == original_answer:
        new_record["teacher_pred"] = target_letter
    elif new_record.get("teacher_pred") == target_letter:
        new_record["teacher_pred"] = original_answer

    if new_record.get("student_pred") == original_answer:
        new_record["student_pred"] = target_letter
    elif new_record.get("student_pred") == target_letter:
        new_record["student_pred"] = original_answer
        
    new_record["augmented"] = True
    new_record["original_answer"] = original_answer
    
    return new_record


# Split dataset by valid letter counts
binary_records = [r for r in data if len(r["valid_letters"]) == 2]
multi_records = [r for r in data if len(r["valid_letters"]) > 2]

# --- 1. Handle Binary Records (Target: 31 A, 32 B) ---
binary_records.sort(key=lambda x: x["id"])
bin_a = [r for r in binary_records if r["answer"] == "A"]
bin_b = [r for r in binary_records if r["answer"] == "B"]

to_swap = bin_a[:30] # We need to move exactly 30 from A to B
keep_a = bin_a[30:]  # 31 A's will stay as A

new_binary = []
failed_binary = 0
for r in to_swap:
    res = permute_record(r, "B")
    if res: 
        new_binary.append(res)
    else: 
        keep_a.append(r)
        failed_binary += 1

final_binary = keep_a + bin_b + new_binary

# --- 2. Handle Multi Records (Target: 11 C, 11 D, 11 E) ---
multi_records.sort(key=lambda x: x["id"])

target_counts = {"C": 11, "D": 11, "E": 11}
final_multi = []
failed_multi = 0

for r in multi_records:
    valid = r["valid_letters"]
    
    # We greedily assign E first, then D, then C, because E is the most restricted (some only have 4 options)
    possible_targets = [l for l in ["E", "D", "C"] if l in valid and target_counts[l] > 0]
    
    if possible_targets:
        target = possible_targets[0]
    else:
        # Fallback if somehow quotas are full (won't happen with 33 records)
        possible_targets = [l for l in ["E", "D", "C"] if l in valid]
        target = possible_targets[0]
        
    res = permute_record(r, target)
    if res:
        final_multi.append(res)
        if target in target_counts:
            target_counts[target] -= 1
    else:
        final_multi.append(r)
        failed_multi += 1

# Merge and save
final_dataset = final_binary + final_multi
final_dataset.sort(key=lambda x: x["id"])

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, indent=2)

print(f"Total records: {len(final_dataset)}")
print(f"Failed to permute: {failed_binary + failed_multi}")
print("\nFinal Rebalanced Distribution:")
label_counts = Counter(r["answer"] for r in final_dataset)
for letter in ["A", "B", "C", "D", "E", "F"]:
    if letter in label_counts:
        pct = 100 * label_counts[letter] / len(final_dataset)
        print(f"  {letter} : {label_counts[letter]:>2} ({pct:.1f}%)")

import json
import re
import copy
from collections import Counter

file_path = "bbh_golden_final_with_conf.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Backup the original dataset just in case
with open("bbh_golden_final_with_conf.bak.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

def swap_options_in_block(block: str, from_letter: str, to_letter: str, valid_letters: list):
    lines = block.split('\n')
    option_map = {}  # letter -> (line_index, option_text, format)

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
    if target_letter not in valid_letters:
        return None

    if "Question\n" not in record["student_prompt"]:
        return None

    prefix_s, q_block = record["student_prompt"].rsplit("Question\n", 1)
    
    new_q_block, success = swap_options_in_block(q_block, "A", target_letter, valid_letters)
    if not success:
        return None

    new_student_prompt = prefix_s + "Question\n" + new_q_block
    
    if "Question\n" + q_block not in record["teacher_prompt"]:
        return None
        
    new_teacher_prompt = record["teacher_prompt"].replace("Question\n" + q_block, "Question\n" + new_q_block, 1)

    new_record = copy.deepcopy(record)
    new_record["teacher_prompt"] = new_teacher_prompt
    new_record["student_prompt"] = new_student_prompt
    
    # Update answer
    new_record["answer"] = target_letter
    
    # Update teacher_pred (was originally A, now target_letter)
    if new_record.get("teacher_pred") == "A":
        new_record["teacher_pred"] = target_letter
    elif new_record.get("teacher_pred") == target_letter:
        new_record["teacher_pred"] = "A"

    # Update student_pred (swap A and target_letter)
    if new_record.get("student_pred") == "A":
        new_record["student_pred"] = target_letter
    elif new_record.get("student_pred") == target_letter:
        new_record["student_pred"] = "A"
        
    new_record["augmented"] = True
    new_record["original_answer"] = "A"
    
    return new_record

a_records = [r for r in data if r["answer"] == "A"]
non_a_records = [r for r in data if r["answer"] != "A"]

# Sort a_records to be deterministic instead of random
a_records.sort(key=lambda x: x["id"])

to_permute = a_records[:30]
keep_as_a = a_records[30:]

TARGET_CYCLE = ["B", "C", "D", "E"]
new_records = []
failed = 0

for cycle_idx, rec in enumerate(to_permute):
    valid = rec["valid_letters"]
    # We can only swap to letters that are actually valid choices for this task
    valid_targets = [l for l in TARGET_CYCLE if l in valid]
    if not valid_targets:
        valid_targets = ["B"]
        
    target = valid_targets[cycle_idx % len(valid_targets)]
    result = permute_record(rec, target)
    
    if result is not None:
        new_records.append(result)
    else:
        keep_as_a.append(rec)
        failed += 1

final_golden = non_a_records + keep_as_a + new_records
final_golden.sort(key=lambda x: x["id"])

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(final_golden, f, indent=2)

print(f"Total records in dataset: {len(final_golden)}")
print(f"Failed to permute (formatting mismatch): {failed}")
print("\nNew Label Distribution:")
label_counts = Counter(r["answer"] for r in final_golden)
for letter in ["A", "B", "C", "D", "E", "F"]:
    if letter in label_counts:
        pct = 100 * label_counts[letter] / len(final_golden)
        print(f"  {letter} : {label_counts[letter]:>3} ({pct:.1f}%)")

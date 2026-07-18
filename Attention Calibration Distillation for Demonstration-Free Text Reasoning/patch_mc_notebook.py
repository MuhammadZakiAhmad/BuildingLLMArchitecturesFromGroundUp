import json
import os

old_data = {
  1: {"options": "Options:\nA) Alice\nB) Anna\nC) David\nD) Luke", "answer": "B"},
  2: {"options": "Options:\nA) Anna\nB) Emma\nC) Mary\nD) Peter", "answer": "C"},
  3: {"options": "Options:\nA) (0,4)\nB) (0,6)\nC) (3,2)\nD) (6,4)", "answer": "D"},
  4: {"options": "Options:\nA) East\nB) North\nC) South\nD) West", "answer": "A"},
  5: {"options": "Options:\nA) Invalid\nB) Valid", "answer": "A"},
  6: {"options": "Options:\nA) No\nB) Yes", "answer": "B"},
  7: {"options": "Options:\nA) False\nB) True", "answer": "B"},
  8: {"options": "Options:\nA) 15\nB) 16\nC) 18\nD) 24", "answer": "C"},
  9: {"options": "Options:\nA) Balanced\nB) Unbalanced", "answer": "A"},
  10: {"options": "Options:\nA) apple chair monkey violin\nB) chair monkey violin apple\nC) violin monkey chair apple\nD) violin monkey chair apple", "answer": "A"},
  11: {"options": "Options:\nA) Alice\nB) Anna\nC) Carl\nD) Mary", "answer": "B"},
  12: {"options": "Options:\nA) (0,6)\nB) (2,4)\nC) (3,2)\nD) (6,1)", "answer": "B"},
  13: {"options": "Options:\nA) Invalid\nB) Valid", "answer": "A"},
  14: {"options": "Options:\nA) No\nB) Yes", "answer": "B"},
  15: {"options": "Options:\nA) False\nB) True", "answer": "B"},
  16: {"options": "Options:\nA) 14\nB) 24\nC) 36\nD) 9", "answer": "C"},
  17: {"options": "Options:\nA) Balanced\nB) Unbalanced", "answer": "A"},
  18: {"options": "Options:\nA) airplane bottle computer window\nB) bottle computer window airplane\nC) window computer bottle airplane\nD) window computer bottle airplane", "answer": "A"},
  19: {"options": "Options:\nA) Friday\nB) Monday\nC) Saturday\nD) Sunday\nE) Thursday\nF) Tuesday\nG) Wednesday", "answer": "E"},
  20: {"options": "Options:\nA) Alice\nB) John\nC) Peter\nD) Sam", "answer": "C"}
}

file_path = 'Phase_2_Reasoning_Tasks.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract existing tasks from the notebook itself since we can't load the JSON files
existing_tasks = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if any('raw_tasks_json = r"""[' in line for line in cell['source']):
            # Find the string definition
            lines = cell['source']
            for i, line in enumerate(lines):
                if 'raw_tasks_json = r"""[' in line:
                    start_idx = i
                    break
            
            json_str = ""
            for line in lines[start_idx:]:
                json_str += line
                if ']"""' in line:
                    break
            
            json_str = json_str.replace('raw_tasks_json = r"""', '').replace('"""\n', '').strip()
            existing_tasks = json.loads(json_str)
            break

mc_tasks = []
for t in existing_tasks:
    tid = t['id']
    old = old_data[tid]
    
    tp = t['teacher_prompt']
    q_idx = tp.rfind("Question\n")
    if q_idx != -1:
        base_examples = tp[:q_idx].strip()
        question_part = tp[q_idx:].replace("Answer:", "").strip()
    else:
        base_examples = ""
        question_part = tp.replace("Answer:", "").strip()
        
    new_tp = base_examples + "\n\n" + question_part + "\n" + old['options'] + "\nAnswer:"
    
    sp = t['student_prompt'].replace("Answer:", "").strip()
    new_sp = sp + "\n" + old['options'] + "\nAnswer:"
    
    mc_tasks.append({
        "id": tid,
        "task": t['task'],
        "teacher_prompt": new_tp,
        "student_prompt": new_sp,
        "answer": old['answer']
    })

all_tasks_str = json.dumps(mc_tasks, indent=2)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if any('raw_tasks_json =' in line for line in cell['source']):
            cell['source'] = [
                'import json\n',
                'raw_tasks_json = r"""' + all_tasks_str + '"""\n',
                '\n',
                'tasks = json.loads(raw_tasks_json)\n',
                'print(f"Loaded {len(tasks)} multiple-choice tasks directly from variable.")\n'
            ]
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Generated multiple choice tasks and patched notebook.")

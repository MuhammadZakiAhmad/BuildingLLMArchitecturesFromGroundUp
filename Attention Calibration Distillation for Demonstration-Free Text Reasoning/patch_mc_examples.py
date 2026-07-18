import json
import random
import re

file_path = 'Phase_2_Reasoning_Tasks.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if any('raw_tasks_json = r"""[' in line for line in cell['source']):
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

def generate_distractors(ans, task_type):
    ans = ans.strip()
    if task_type in ['formal_fallacies']:
        return ['Invalid', 'Valid']
    if task_type in ['logical_deduction']:
        return ['No', 'Yes']
    if task_type in ['boolean_expressions', 'dyck_language']:
        if ans in ['True', 'False']: return ['False', 'True']
        if ans in ['Balanced', 'Unbalanced']: return ['Balanced', 'Unbalanced']
    if task_type == 'navigate':
        if '(' in ans:
            opts = list(set([ans, '(0,0)', '(1,2)', '(3,4)', '(2,1)', '(5,5)', '(6,6)']))
            opts.remove(ans)
            return sorted([ans] + opts[:3])
        else:
            return sorted(['North', 'South', 'East', 'West'])
    if task_type == 'tracking_shuffled_objects':
        names = ['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'John', 'Mary', 'Peter', 'Sam', 'Tom', 'Luke', 'Anna', 'Carl', 'Jack', 'Kate', 'Ryan', 'Amy', 'Ben', 'Eva']
        opts = list(set([ans] + random.sample(names, 10)))
        opts.remove(ans)
        return sorted([ans] + opts[:3])
    if task_type == 'date_understanding':
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return sorted(days)
    if task_type == 'multistep_arithmetic':
        try:
            val = int(ans)
            return sorted([str(val), str(val+1), str(val-1), str(val+2)])
        except:
            return sorted([ans, '10', '15', '20'])
    if task_type == 'word_sorting':
        words = ans.split()
        opts = [ans]
        for _ in range(15):
            w2 = words.copy()
            random.shuffle(w2)
            j = " ".join(w2)
            if j not in opts: opts.append(j)
        return sorted(opts[:4])
    return sorted([ans, 'A', 'B', 'C'])

for t in existing_tasks:
    tp = t['teacher_prompt']
    
    if "Example 1" not in tp:
        continue
        
    parts = tp.split("Example ")
    new_tp = parts[0]
    
    for part in parts[1:]:
        ans_match = re.search(r"Answer:\s*(.+?)(?:\n|$)", part)
        if ans_match and (part.startswith("1\n") or part.startswith("2\n") or part.startswith("3\n") or part.startswith("4\n")):
            ans_val = ans_match.group(1).strip()
            
            opts = generate_distractors(ans_val, t['task'])
            letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            correct_idx = opts.index(ans_val)
            correct_letter = letters[correct_idx]
            
            opts_str = "Options:\n" + "\n".join([f"{letters[i]}) {opts[i]}" for i in range(len(opts))])
            
            rest_of_part = part[ans_match.end():].strip()
            if rest_of_part:
                rest_of_part = "\n\n" + rest_of_part
                
            new_part = part[:ans_match.start()] + opts_str + "\nAnswer: " + correct_letter + rest_of_part + "\n\n"
            new_tp += "Example " + new_part
        else:
            new_tp += "Example " + part
            
    t['teacher_prompt'] = new_tp.strip()

all_tasks_str = json.dumps(existing_tasks, indent=2)

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

print("Teacher examples rewritten to multiple choice successfully.")

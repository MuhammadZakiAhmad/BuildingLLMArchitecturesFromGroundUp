import json

file_path = 'Phase_2_Reasoning_Tasks.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── 1. Fix the A) 4 distractor bug in the tasks data ──
# Find the raw_tasks_json cell and reload the tasks, then fix bad options
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if 'raw_tasks_json = r"""' in src:
            start = src.index('r"""') + 4
            end = src.rindex('"""')
            raw_json = src[start:end]
            tasks = json.loads(raw_json)
            
            # Fix any MC option that is a single digit/number for non-arithmetic tasks
            fixed = 0
            names_pool = ['Alice','Bob','Charlie','David','Emma','Jack','Kate','Luke',
                          'Mary','Peter','Amy','Ben','Carl','Eva','John','Ryan','Sam',
                          'Tom','Anna','Sarah','Mike','Lucy','Omar','Nora','Zara','Leo',
                          'Mia','Kai','Finn','Ivy']
            coord_pool = ['(0,0)','(1,2)','(2,3)','(3,4)','(4,5)','(5,6)',
                          '(0,-1)','(1,-2)','(2,-3)','(-1,0)','(-2,1)']
            
            for t in tasks:
                # Find options block in teacher_prompt
                for prompt_key in ['teacher_prompt', 'student_prompt']:
                    p = t[prompt_key]
                    lines = p.split('\n')
                    new_lines = []
                    in_opts = False
                    for line in lines:
                        if line.startswith('Options:'):
                            in_opts = True
                        elif in_opts and line.startswith(('A)','B)','C)','D)')):
                            val = line[3:].strip()
                            # Bad: single digit or very short pure number
                            if val.lstrip('-').isdigit() and t['task'] != 'multistep_arithmetic':
                                # Replace with a valid name
                                replacement = [n for n in names_pool 
                                               if n not in p and n != t['answer']]
                                if replacement:
                                    line = line[:3] + replacement[0]
                                    fixed += 1
                        elif in_opts and not line.startswith(('A)','B)','C)','D)')) and line.strip():
                            in_opts = False
                        new_lines.append(line)
                    t[prompt_key] = '\n'.join(new_lines)
            
            print(f"Fixed {fixed} bad distractor options")
            
            # Rewrite the cell
            new_tasks_str = json.dumps(tasks, indent=2)
            cell['source'] = [
                'import json\n',
                'raw_tasks_json = r"""' + new_tasks_str + '"""\n',
                '\n',
                'tasks = json.loads(raw_tasks_json)\n',
                'print(f"Loaded {len(tasks)} tasks.")\n'
            ]
            break

# ── 2. Insert golden task summary cell after the golden_tasks summary cell ──
golden_summary_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "## ── Golden Task Breakdown ──\n",
        "print(f'\\n{\"=\"*75}')\n",
        "print(f'{\"ID\":<6} {\"Type\":<30} {\"Correct\":>8} {\"Student\":>9} {\"Teacher\":>9}')\n",
        "print(f'{\"=\"*75}')\n",
        "for r in golden_tasks:\n",
        "    s_mark = '❌'\n",
        "    t_mark = '✅'\n",
        "    print(f\"{r['id']:<6} {r['task']:<30} {r['answer']:>8} \"\n",
        "          f\"{r['student_pred']:>4} {s_mark}   {r['teacher_pred']:>4} {t_mark}\")\n",
        "print(f'{\"=\"*75}')\n",
        "print(f'\\nTotal golden tasks: {len(golden_tasks)}')\n"
    ]
}

# Find the cell at index 14582 area (golden summary cell) and insert after it
target_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if 'golden_tasks = [r for r in results' in src:
            target_idx = i
            break

if target_idx is not None:
    nb['cells'].insert(target_idx + 1, golden_summary_cell)
    print(f"Inserted golden breakdown cell after index {target_idx}")
else:
    nb['cells'].append(golden_summary_cell)
    print("Appended golden breakdown cell at end")

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Done.")

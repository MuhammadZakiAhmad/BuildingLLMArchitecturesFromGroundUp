import json
import os

with open('forsmol.json', 'r', encoding='utf-8') as f:
    tasks1 = json.load(f)

with open('forsmol2.json', 'r', encoding='utf-8') as f:
    tasks2 = json.load(f)

all_tasks = tasks1 + tasks2
all_tasks_str = json.dumps(all_tasks, indent=2)

file_path = 'Phase_2_Reasoning_Tasks.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # we need to match the cell we just modified, or the original one if we reverted
        if any('raw_tasks_json' in line or 'forsmol.json' in line for line in cell['source']):
            cell['source'] = [
                'import json\n',
                'raw_tasks_json = r"""' + all_tasks_str + '"""\n',
                '\n',
                'tasks = json.loads(raw_tasks_json)\n',
                'print(f"Loaded {len(tasks)} tasks directly from variable.")\n'
            ]
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook patched to embed JSON string.')

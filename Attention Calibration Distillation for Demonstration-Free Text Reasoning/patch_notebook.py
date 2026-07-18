import json
import os

file_path = 'Phase_2_Reasoning_Tasks.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if any('raw_tasks_json = r"""[' in line for line in cell['source']):
            cell['source'] = [
                "import json\n",
                "\n",
                "with open('forsmol.json', 'r', encoding='utf-8') as f:\n",
                "    tasks1 = json.load(f)\n",
                "\n",
                "with open('forsmol2.json', 'r', encoding='utf-8') as f:\n",
                "    tasks2 = json.load(f)\n",
                "\n",
                "tasks = tasks1 + tasks2\n",
                "print(f\"Loaded {len(tasks)} tasks from forsmol.json and forsmol2.json.\")\n"
            ]
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook patched successfully.')

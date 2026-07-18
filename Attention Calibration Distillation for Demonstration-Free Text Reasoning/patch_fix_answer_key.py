import json

file_path = 'Phase_2_Reasoning_Tasks.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix the summary cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if "Golden Task Breakdown" in src and "r['answer']" in src:
            cell['source'] = [
                "# ── Golden Task Breakdown ──\n",
                "# Look up true answers from the tasks list (results dict has no 'answer' key)\n",
                "answer_lookup = {t['id']: t['answer'] for t in tasks}\n",
                "\n",
                "print(f'\\n{\"=\"*75}')\n",
                "print(f'{\"ID\":<6} {\"Type\":<30} {\"Correct\":>8} {\"Student\":>9} {\"Teacher\":>9}')\n",
                "print(f'{\"=\"*75}')\n",
                "for r in golden_tasks:\n",
                "    true_ans = answer_lookup.get(r['id'], '?')\n",
                "    print(f\"{r['id']:<6} {r['task']:<30} {true_ans:>8} \"\n",
                "          f\"{r['student_pred']:>4} ❌   {r['teacher_pred']:>4} ✅\")\n",
                "print(f'{\"=\"*75}')\n",
                "print(f'\\nTotal golden tasks: {len(golden_tasks)}')\n"
            ]
            cell['outputs'] = []
            print("Fixed summary cell.")
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Done.")

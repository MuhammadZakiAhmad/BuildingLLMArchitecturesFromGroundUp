import json

file_path = 'Phase_2_Reasoning_Tasks.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── New summary cell to add after the evaluation cell ──
summary_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Golden Task Summary ──\n",
        "print(f'\\nGolden Task IDs: {golden_ids}')\n",
        "print(f'\\n{\"=\"*72}')\n",
        "print(f'{\"ID\":<6} {\"Type\":<30} {\"True\":>6} {\"Student\":>8} {\"Teacher\":>8}')\n",
        "print(f'{\"=\"*72}')\n",
        "for r in results:\n",
        "    if r['id'] in golden_ids:\n",
        "        print(f\"{r['id']:<6} {r['task']:<30} {r['answer']:>6} \"\n",
        "              f\"{r['s_pred']:>8} {r['t_pred']:>8}\")\n",
        "print(f'{\"=\"*72}')\n",
        "print(f'\\nLegend: True=correct answer | Student=0-shot pred | Teacher=4-shot pred')\n",
        "print(f'Golden = Teacher ✅ and Student ❌ on the same task')\n"
    ]
}

# ── Find the evaluation cell and insert the summary cell right after it ──
eval_cell_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if 'GOLDEN TASKS' in src and 'golden_ids' in src:
            eval_cell_idx = i
            break

# Also fix the evaluation cell to store results list for the summary cell
if eval_cell_idx is not None:
    eval_src = "".join(nb['cells'][eval_cell_idx]['source'])
    
    # Check if results list is already being built
    if 'results = []' not in eval_src:
        # Patch the eval cell to accumulate results
        new_src = eval_src.replace(
            "golden_ids = []",
            "golden_ids = []\nresults = []"
        )
        # Add result appending inside loop after we know s_correct and t_correct
        # Find the line that appends to golden_ids and add results append before it
        new_src = new_src.replace(
            "if t_correct and not s_correct:",
            "results.append({'id': t['id'], 'task': t['task'], 'answer': t['answer'], 's_pred': s_pred, 't_pred': t_pred})\n    if t_correct and not s_correct:"
        )
        nb['cells'][eval_cell_idx]['source'] = list(new_src)

    # Insert summary cell right after eval cell
    nb['cells'].insert(eval_cell_idx + 1, summary_cell)
    print(f"Inserted summary cell after eval cell at index {eval_cell_idx}")
else:
    print("WARNING: eval cell not found by GOLDEN TASKS marker, appending summary cell at end")
    nb['cells'].append(summary_cell)

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Done.")

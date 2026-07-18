import json

file_path = 'Phase_2_Reasoning_Tasks.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if any('AutoModelForCausalLM.from_pretrained' in line for line in cell['source']):
            new_source = []
            for line in cell['source']:
                if 'tokenizer = AutoTokenizer.from_pretrained' in line:
                    new_source.extend([
                        "if 'tokenizer' not in globals():\n",
                        "    tokenizer = AutoTokenizer.from_pretrained(model_name)\n"
                    ])
                elif 'model = AutoModelForCausalLM.from_pretrained' in line:
                    new_source.extend([
                        "if 'model' not in globals():\n",
                        "    model = AutoModelForCausalLM.from_pretrained(\n"
                    ])
                elif '    model_name,' in line or '    torch_dtype' in line or '    attn_implementation' in line or ').to(device)' in line:
                    new_source.append("    " + line) # Indent it to be under the if block
                else:
                    new_source.append(line)
            cell['source'] = new_source
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Added conditional loading to prevent OOM.')

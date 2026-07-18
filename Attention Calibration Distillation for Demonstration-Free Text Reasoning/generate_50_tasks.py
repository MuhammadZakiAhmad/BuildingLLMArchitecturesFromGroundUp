import json
import random

NAVIGATE_EXAMPLES = """Task: Determine the final coordinates.

Example 1
Start (0,0)
North 2
East 3
Options:
A) (2,1)
B) (3,2)
C) (3,4)
D) (5,5)
Answer: B

Example 2
Start (4,1)
West 2
South 1
Options:
A) (2,0)
B) (2,1)
C) (3,4)
D) (5,5)
Answer: A

Example 3
Start (1,5)
East 4
South 2
Options:
A) (2,1)
B) (3,4)
C) (5,3)
D) (5,5)
Answer: C

Example 4
Start (3,3)
North 1
West 3
Options:
A) (0,4)
B) (2,1)
C) (3,4)
D) (5,5)
Answer: A"""

DYCK_EXAMPLES = """Task: Determine whether the parentheses are Balanced or Unbalanced.

Example 1
(()())
Options:
A) Balanced
B) Unbalanced
Answer: A

Example 2
((())
Options:
A) Balanced
B) Unbalanced
Answer: B

Example 3
(((())))
Options:
A) Balanced
B) Unbalanced
Answer: A

Example 4
())(()
Options:
A) Balanced
B) Unbalanced
Answer: B"""

WORD_SORTING_EXAMPLES = """Task: Sort the words alphabetically.

Example 1
pear apple banana
Options:
A) apple banana pear
B) apple pear banana
C) banana apple pear
D) pear apple banana
Answer: A

Example 2
dog cat bird
Options:
A) bird cat dog
B) cat dog bird
C) dog bird cat
D) dog cat bird
Answer: A

Example 3
orange kiwi grape
Options:
A) grape kiwi orange
B) grape orange kiwi
C) orange grape kiwi
D) orange kiwi grape
Answer: A

Example 4
zebra lion tiger
Options:
A) lion tiger zebra
B) lion zebra tiger
C) tiger zebra lion
D) zebra tiger lion
Answer: A"""

WORDS = ['apple', 'banana', 'cherry', 'desk', 'chair', 'table', 'lion', 'tiger', 'zebra', 'monkey', 'bird', 'dog', 'cat', 'violin', 'guitar', 'piano', 'orange', 'kiwi', 'grape', 'window', 'door', 'computer', 'bottle', 'airplane', 'car', 'train', 'shoe', 'shirt', 'hat']

def build_navigate():
    x, y = random.randint(0, 5), random.randint(0, 5)
    start = f"Start ({x},{y})"
    steps = random.randint(3, 4)
    dirs = ['North', 'South', 'East', 'West']
    
    q_lines = [start]
    for _ in range(steps):
        d = random.choice(dirs)
        dist = random.randint(1, 4)
        q_lines.append(f"{d} {dist}")
        if d == 'North': y += dist
        elif d == 'South': y -= dist
        elif d == 'East': x += dist
        elif d == 'West': x -= dist
    
    ans_str = f"({x},{y})"
    
    opts = set([ans_str])
    while len(opts) < 4:
        opts.add(f"({x+random.randint(-2,2)},{y+random.randint(-2,2)})")
    
    opts = sorted(list(opts))
    correct_letter = ['A', 'B', 'C', 'D'][opts.index(ans_str)]
    
    q_text = "Question\n" + "\n".join(q_lines)
    opts_str = "Options:\n" + "\n".join([f"{['A','B','C','D'][i]}) {opts[i]}" for i in range(4)])
    
    return q_text, opts_str, correct_letter

def generate_balanced(length):
    if length == 0: return ""
    if length == 2: return "()"
    r = random.random()
    if r < 0.5:
        return "(" + generate_balanced(length-2) + ")"
    else:
        split = random.choice(range(2, length, 2))
        return generate_balanced(split) + generate_balanced(length-split)

def build_dyck():
    length = random.choice([6, 8, 10])
    is_balanced = random.choice([True, False])
    
    if is_balanced:
        s = generate_balanced(length)
    else:
        s = generate_balanced(length)
        pos = random.randint(0, len(s)-1)
        l = list(s)
        l[pos] = '(' if l[pos] == ')' else ')'
        s = "".join(l)
    
    ans_str = "Balanced" if is_balanced else "Unbalanced"
    correct_letter = "A" if is_balanced else "B"
    
    q_text = f"Question\n{s}"
    opts_str = "Options:\nA) Balanced\nB) Unbalanced"
    
    return q_text, opts_str, correct_letter

def build_word_sorting():
    w = random.sample(WORDS, random.randint(4, 5))
    ans_str = " ".join(sorted(w))
    
    q_text = "Question\n" + " ".join(w)
    
    opts = [ans_str]
    for _ in range(15):
        w2 = w.copy()
        random.shuffle(w2)
        j = " ".join(w2)
        if j not in opts: opts.append(j)
    
    opts = sorted(opts[:4])
    correct_letter = ['A', 'B', 'C', 'D'][opts.index(ans_str)]
    opts_str = "Options:\n" + "\n".join([f"{['A','B','C','D'][i]}) {opts[i]}" for i in range(4)])
    
    return q_text, opts_str, correct_letter

tasks = []
task_id = 1

for _ in range(17):
    q_text, opts_str, correct_letter = build_navigate()
    sp = "Task: Determine the final coordinates.\n\n" + q_text + "\n" + opts_str + "\nAnswer:"
    tp = NAVIGATE_EXAMPLES + "\n\n" + q_text + "\n" + opts_str + "\nAnswer:"
    tasks.append({"id": task_id, "task": "navigate", "teacher_prompt": tp, "student_prompt": sp, "answer": correct_letter})
    task_id += 1

for _ in range(17):
    q_text, opts_str, correct_letter = build_dyck()
    sp = "Task: Determine whether the parentheses are Balanced or Unbalanced.\n\n" + q_text + "\n" + opts_str + "\nAnswer:"
    tp = DYCK_EXAMPLES + "\n\n" + q_text + "\n" + opts_str + "\nAnswer:"
    tasks.append({"id": task_id, "task": "dyck_language", "teacher_prompt": tp, "student_prompt": sp, "answer": correct_letter})
    task_id += 1

for _ in range(16):
    q_text, opts_str, correct_letter = build_word_sorting()
    sp = "Task: Sort the words alphabetically.\n\n" + q_text + "\n" + opts_str + "\nAnswer:"
    tp = WORD_SORTING_EXAMPLES + "\n\n" + q_text + "\n" + opts_str + "\nAnswer:"
    tasks.append({"id": task_id, "task": "word_sorting", "teacher_prompt": tp, "student_prompt": sp, "answer": correct_letter})
    task_id += 1

all_tasks_str = json.dumps(tasks, indent=2)

file_path = 'Phase_2_Reasoning_Tasks.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if any('raw_tasks_json =' in line for line in cell['source']):
            cell['source'] = [
                'import json\n',
                'raw_tasks_json = r"""' + all_tasks_str + '"""\n',
                '\n',
                'tasks = json.loads(raw_tasks_json)\n',
                'print(f"Loaded {len(tasks)} newly generated Goldilocks tasks.")\n'
            ]
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Generated 50 Goldilocks tasks and injected into notebook.")

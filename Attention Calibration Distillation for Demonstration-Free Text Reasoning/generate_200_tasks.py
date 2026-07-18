import json
import random

random.seed(99)

LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# ─────────────────────────────────────────────
# 2-SHOT teacher blocks (trimmed from 4 → 2 to
# reduce context confusion on the 1.7B model)
# ─────────────────────────────────────────────

TP_TRACKING_2P = """Task: Track the ownership of objects after each swap.

Example 1
Alice has the red ball.
Bob has the blue ball.
Swap Alice and Bob.
Who has the red ball?
Options:
A) Bob
B) Emma
C) Eva
D) Ryan
Answer: A

Example 2
Tom has the key.
Emma has the coin.
Swap Tom and Emma.
Who has the coin?
Options:
A) Alice
B) Bob
C) Emma
D) Tom
Answer: D"""

TP_TRACKING_3P = """Task: Track the ownership of objects after multiple swaps.

Example 1
Alice has the ball.
Bob has the key.
Charlie has the book.
Swap Alice and Bob.
Swap Bob and Charlie.
Who has the ball?
Options:
A) Amy
B) Carl
C) Charlie
D) Eva
Answer: C

Example 2
Tom has the coin.
Sam has the ring.
Eva has the pen.
Swap Tom and Eva.
Swap Eva and Sam.
Who has the coin?
Options:
A) Bob
B) Emma
C) John
D) Sam
Answer: D"""

TP_NAVIGATE_COORD = """Task: Determine the final coordinates.

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
Answer: A"""

TP_BOOLEAN = """Task: Evaluate the Boolean expression.

Example 1
(True AND False) OR True
Options:
A) False
B) True
Answer: B

Example 2
(False OR False) AND True
Options:
A) False
B) True
Answer: A"""

TP_DYCK = """Task: Determine whether the parentheses are Balanced or Unbalanced.

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
Answer: B"""

TP_ARITHMETIC = """Task: Solve the arithmetic expression.

Example 1
(4+2)*3
Options:
A) 17
B) 18
C) 19
D) 20
Answer: B

Example 2
10-(3*2)
Options:
A) 3
B) 4
C) 5
D) 6
Answer: B"""

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

NAMES = ['Alice','Bob','Charlie','David','Emma','Jack','Kate','Luke','Mary','Peter',
         'Amy','Ben','Carl','Eva','John','Ryan','Sam','Tom','Anna','Sarah','Mike','Lucy',
         'Omar','Nora','Zara','Leo','Mia','Kai','Finn','Ivy']
OBJS  = ['ball','key','book','coin','pen','ring','hat','bag','map','phone',
         'watch','wallet','laptop','tablet','camera','notebook','guitar','doll',
         'cube','mirror','flask','badge','glove','lamp','mask','shell','sword','crown']
WORDS = ['apple','banana','cherry','desk','chair','table','lion','tiger','zebra',
         'monkey','bird','dog','cat','guitar','piano','orange','kiwi','grape',
         'window','door','computer','bottle','airplane','car','train','shoe',
         'shirt','hat','river','cloud','stone','flame','sword','crown','tower',
         'forest','ocean','bridge','arrow','feather']

def mc4(ans, pool):
    """Return (opts_str, letter) for 4 MC options."""
    opts = sorted(list({ans} | set(random.sample([x for x in pool if x != ans],
                                                 min(3, len([x for x in pool if x != ans])))))) 
    # ensure exactly 4 and ans is in there
    while len(opts) < 4:
        opts.append(str(random.randint(0,9)))
    opts = sorted(list(set(opts)))[:4]
    if ans not in opts:
        opts[-1] = ans
        opts = sorted(opts)
    letter = LETTERS[opts.index(ans)]
    opts_str = "Options:\n" + "\n".join(f"{LETTERS[i]}) {opts[i]}" for i in range(len(opts)))
    return opts_str, letter

def task_entry(tid, task_type, teacher_block, q_text, opts_str, letter):
    tp = teacher_block + "\n\nQuestion\n" + q_text + "\n" + opts_str
    header = teacher_block.split("\n")[0]  # "Task: ..."
    sp = header + "\n\nQuestion\n" + q_text + "\n" + opts_str
    return {"id": tid, "task": task_type,
            "teacher_prompt": tp, "student_prompt": sp, "answer": letter}

# ─────────────────────────────────────────────
# Generators
# ─────────────────────────────────────────────

def gen_tracking_2p(tid):
    names = random.sample(NAMES, 2)
    objs  = random.sample(OBJS,  2)
    n1,n2 = names; o1,o2 = objs
    # after 1 swap: n1 has o2, n2 has o1
    ask  = random.choice([o1, o2])
    ans  = n2 if ask == o1 else n1
    distractors = [x for x in NAMES if x not in names]
    pool = sorted(list(set([ans, names[0], names[1]] + random.sample(distractors, 1))))[:4]
    opts_str, letter = mc4(ans, pool)
    q = f"{n1} has the {o1}.\n{n2} has the {o2}.\nSwap {n1} and {n2}.\nWho has the {ask}?"
    return task_entry(tid, "tracking_shuffled_objects", TP_TRACKING_2P, q, opts_str, letter)

def gen_tracking_3p(tid):
    names = random.sample(NAMES, 3)
    objs  = random.sample(OBJS,  3)
    n0,n1,n2 = names; o0,o1,o2 = objs

    # pick 2 distinct swaps
    swap_pairs = [(0,1),(0,2),(1,2)]
    sw1, sw2 = random.sample(swap_pairs, 2)

    current_owners = {o0:n0, o1:n1, o2:n2}
    name_objs = {n0:o0, n1:o1, n2:o2}

    def do_swap(na, nb, name_objs):
        oa, ob = name_objs[na], name_objs[nb]
        name_objs[na], name_objs[nb] = ob, oa

    na1, nb1 = names[sw1[0]], names[sw1[1]]
    na2, nb2 = names[sw2[0]], names[sw2[1]]
    do_swap(na1, nb1, name_objs)
    do_swap(na2, nb2, name_objs)

    ask_obj = random.choice(objs)
    ans = [n for n,o in name_objs.items() if o == ask_obj][0]
    pool = sorted(list(set(names + random.sample([x for x in NAMES if x not in names], 1))))[:4]
    if ans not in pool: pool[-1] = ans; pool = sorted(pool)
    opts_str, letter = mc4(ans, pool)
    q = (f"{n0} has the {o0}.\n{n1} has the {o1}.\n{n2} has the {o2}.\n"
         f"Swap {na1} and {nb1}.\nSwap {na2} and {nb2}.\nWho has the {ask_obj}?")
    return task_entry(tid, "tracking_shuffled_objects", TP_TRACKING_3P, q, opts_str, letter)

def gen_navigate_coord(tid):
    x,y = random.randint(0,6), random.randint(0,6)
    steps = random.randint(3,5)
    dirs = ['North','South','East','West']
    lines = [f"Start ({x},{y})"]
    for _ in range(steps):
        d = random.choice(dirs); dist = random.randint(1,5)
        lines.append(f"{d} {dist}")
        if d=='North': y+=dist
        elif d=='South': y-=dist
        elif d=='East': x+=dist
        elif d=='West': x-=dist
    ans = f"({x},{y})"
    pool = {ans}
    while len(pool) < 4:
        dx,dy = random.randint(-3,3), random.randint(-3,3)
        candidate = f"({x+dx},{y+dy})"
        if candidate != ans: pool.add(candidate)
    pool = sorted(list(pool))[:4]
    if ans not in pool: pool[-1]=ans; pool=sorted(pool)
    opts_str = "Options:\n" + "\n".join(f"{LETTERS[i]}) {pool[i]}" for i in range(4))
    letter = LETTERS[pool.index(ans)]
    q = "\n".join(lines)
    return task_entry(tid, "navigate", TP_NAVIGATE_COORD, q, opts_str, letter)

BOOL_EXPRS = [
    ("((True OR False) AND False) OR (False OR True)", True),
    ("(True AND (False OR True)) OR (False AND False)", True),
    ("(False AND True) OR (False OR False)", False),
    ("(True OR True) AND (False AND True)", False),
    ("((False OR True) AND True) OR False", True),
    ("(True AND True) AND (True OR False)", True),
    ("(False OR False) AND (True OR True)", False),
    ("((True AND False) OR True) AND True", True),
    ("(False AND False) OR (False OR False)", False),
    ("(True OR False) AND (True AND False)", False),
    ("NOT (True AND False)", True),
    ("NOT (False OR False)", True),
    ("(True AND NOT False) OR False", True),
    ("(False OR NOT True) AND True", False),
    ("NOT ((True AND True) AND False)", True),
    ("(NOT False) AND (NOT True)", False),
    ("(True OR False) AND NOT (False AND True)", True),
    ("NOT (False OR True) OR True", True),
    ("(NOT True) OR (NOT False)", True),
    ("NOT ((False OR False) OR (True AND False))", True),
    ("(True AND False) OR (NOT True)", False),
    ("NOT (True OR False) AND True", False),
    ("(False AND True) OR NOT (False AND True)", True),
    ("(NOT False OR False) AND True", True),
    ("NOT (True AND (False OR True))", False),
    ("(True OR False) OR (False AND False)", True),
    ("(False AND True) AND (True OR False)", False),
    ("NOT False AND NOT True", False),
    ("(True OR True) OR (False AND True)", True),
    ("NOT (True OR True) AND False", False),
    ("(True AND True) OR NOT (True AND True)", True),
    ("(False OR False) OR NOT (True OR True)", False),
    ("NOT (False AND False) AND (True OR False)", True),
    ("(True AND False) OR NOT (True AND False)", True),
    ("NOT (True OR False) OR NOT (False AND True)", True),
    ("(False OR NOT True) OR (True AND NOT False)", True),
    ("NOT (True AND True) OR (False AND False)", False),
    ("(NOT True AND NOT False) OR True", True),
    ("NOT (False OR NOT False) AND True", False),
    ("(True AND NOT True) OR NOT (False OR False)", True),
]

def gen_boolean(tid, expr_pool):
    expr_text, result = random.choice(expr_pool)
    ans = "B" if result else "A"
    opts_str = "Options:\nA) False\nB) True"
    return task_entry(tid, "boolean_expressions", TP_BOOLEAN, expr_text, opts_str, ans)

def gen_balanced_str(n):
    if n == 0: return ""
    if n == 2: return "()"
    if random.random() < 0.5: return "(" + gen_balanced_str(n-2) + ")"
    split = random.choice(range(2, n, 2))
    return gen_balanced_str(split) + gen_balanced_str(n-split)

def gen_dyck(tid):
    length = random.choice([6, 8, 10, 12])
    is_bal = random.choice([True, False])
    s = gen_balanced_str(length)
    if not is_bal:
        pos = random.randint(0, len(s)-1)
        l = list(s); l[pos] = '(' if l[pos]==')' else ')'; s = "".join(l)
    ans = "A" if is_bal else "B"
    opts_str = "Options:\nA) Balanced\nB) Unbalanced"
    return task_entry(tid, "dyck_language", TP_DYCK, s, opts_str, ans)

ARITH_EXPRS = [
    ("((12/3)+5)*2", 18), ("((20/4)+7)*3", 36), ("((6+2)*3)-4", 20),
    ("18/(3+3)", 3),       ("(15-5)*(8/2)", 40), ("7+(18/3)-2", 11),
    ("((9-3)*4)+2", 26),   ("(100/10)+(3*4)", 22), ("((5+3)*(2+2))-6", 26),
    ("(50/5)-(3*2)", 4),   ("((8+2)*5)-10", 40),  ("(3*(4+5))-7", 20),
    ("(30/6)+(4*3)", 17),  ("((7-2)*(8-3))", 25), ("(2**(3+1))-6", 10),
]

def gen_arithmetic(tid):
    expr_text, ans_val = random.choice(ARITH_EXPRS)
    distractors = sorted(list({ans_val, ans_val+2, ans_val-2, ans_val+4,
                                ans_val-4, ans_val+1, ans_val-1, ans_val+3}))
    opts = sorted(list(set(distractors)))[:4]
    if ans_val not in opts: opts[-1]=ans_val; opts=sorted(opts)
    opts_str = "Options:\n" + "\n".join(f"{LETTERS[i]}) {opts[i]}" for i in range(4))
    letter = LETTERS[opts.index(ans_val)]
    return task_entry(tid, "multistep_arithmetic", TP_ARITHMETIC, expr_text, opts_str, letter)

# ─────────────────────────────────────────────
# Build 200 tasks (weighted by hit rate)
# ─────────────────────────────────────────────
# navigate coord:  80 (40% hit → ~32 golden)
# boolean:         40 (20% hit → ~8 golden)
# dyck:            40 (20% hit → ~8 golden)
# tracking 3p:     20 (10% hit → ~2 golden)
# tracking 2p:     10 (10% hit → ~1 golden)
# arithmetic:      10 (10% hit → ~1 golden)
# Total:          200 → expected ~52 golden

tasks = []
tid = 1

print("Generating navigate coord (80 tasks)...")
for _ in range(80):
    tasks.append(gen_navigate_coord(tid)); tid += 1

print("Generating boolean_expressions (40 tasks)...")
for _ in range(40):
    tasks.append(gen_boolean(tid, BOOL_EXPRS)); tid += 1

print("Generating dyck_language (40 tasks)...")
for _ in range(40):
    tasks.append(gen_dyck(tid)); tid += 1

print("Generating tracking 3-person (20 tasks)...")
for _ in range(20):
    tasks.append(gen_tracking_3p(tid)); tid += 1

print("Generating tracking 2-person (10 tasks)...")
for _ in range(10):
    tasks.append(gen_tracking_2p(tid)); tid += 1

print("Generating multistep_arithmetic (10 tasks)...")
for _ in range(10):
    tasks.append(gen_arithmetic(tid)); tid += 1

# Shuffle so task types are interleaved during evaluation
random.shuffle(tasks)
for i, t in enumerate(tasks): t['id'] = i + 1

from collections import Counter
print(f"\nTotal tasks: {len(tasks)}")
print("Distribution:", dict(Counter(t['task'] for t in tasks)))

# ─────────────────────────────────────────────
# Inject into notebook
# ─────────────────────────────────────────────
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
                'print(f"Loaded {len(tasks)} carefully selected tasks (2-shot teacher).")\n'
            ]
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\nInjected into notebook successfully.")

import json
import random

random.seed(42)

LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# ─────────────────────────────────────────────
# Shared teacher few-shot blocks (proven to work)
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
Answer: D

Example 3
John has the book.
Sarah has the pen.
Swap John and Sarah.
Who has the book?
Options:
A) Bob
B) Emma
C) Ryan
D) Sarah
Answer: D

Example 4
Mike has the watch.
Lucy has the phone.
Swap Mike and Lucy.
Who has the phone?
Options:
A) Amy
B) Bob
C) Mike
D) Ryan
Answer: C"""

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
Answer: D

Example 3
Amy has the hat.
Ben has the bag.
Carl has the map.
Swap Ben and Carl.
Swap Amy and Ben.
Who has the hat?
Options:
A) Amy
B) Bob
C) Carl
D) John
Answer: C

Example 4
Jack has the phone.
Kate has the watch.
Luke has the wallet.
Swap Jack and Luke.
Swap Kate and Luke.
Who has the watch?
Options:
A) Amy
B) Emma
C) Jack
D) John
Answer: C"""

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

TP_NAVIGATE_DIR = """Task: Determine the final direction.

Example 1
Face North.
Turn Right.
Turn Right.
Options:
A) East
B) North
C) South
D) West
Answer: C

Example 2
Face East.
Turn Left.
Options:
A) East
B) North
C) South
D) West
Answer: B

Example 3
Face South.
Turn Left.
Turn Left.
Options:
A) East
B) North
C) South
D) West
Answer: B

Example 4
Face West.
Turn Right.
Options:
A) East
B) North
C) South
D) West
Answer: B"""

TP_FORMAL = """Task: Decide whether the argument is Valid or Invalid.

Example 1
All cats are mammals.
Tom is a cat.
Therefore Tom is a mammal.
Options:
A) Invalid
B) Valid
Answer: B

Example 2
All birds fly.
Tweety flies.
Therefore Tweety is a bird.
Options:
A) Invalid
B) Valid
Answer: A

Example 3
All apples are fruits.
This is a fruit.
Therefore this is an apple.
Options:
A) Invalid
B) Valid
Answer: A

Example 4
All doctors studied medicine.
Alice is a doctor.
Therefore Alice studied medicine.
Options:
A) Invalid
B) Valid
Answer: B"""

TP_LOGICAL = """Task: Answer Yes or No.

Example 1
Every cat is an animal.
Every animal breathes.
Tom is a cat.
Does Tom breathe?
Options:
A) No
B) Yes
Answer: B

Example 2
Every engineer knows math.
Everyone who knows math can solve equations.
Alice is an engineer.
Can Alice solve equations?
Options:
A) No
B) Yes
Answer: B

Example 3
Every whale is a mammal.
Every mammal is warm-blooded.
Blue is a whale.
Is Blue warm-blooded?
Options:
A) No
B) Yes
Answer: B

Example 4
Every bird has feathers.
Every animal with feathers lays eggs.
Robin is a bird.
Does Robin lay eggs?
Options:
A) No
B) Yes
Answer: B"""

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
Answer: A

Example 3
(True OR False) AND (True AND True)
Options:
A) False
B) True
Answer: B

Example 4
(False AND True) OR False
Options:
A) False
B) True
Answer: A"""

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
Answer: B

Example 3
24/6+7
Options:
A) 10
B) 11
C) 12
D) 13
Answer: B

Example 4
(8-3)*(5-2)
Options:
A) 14
B) 15
C) 16
D) 17
Answer: B"""

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

TP_WORD = """Task: Sort the words alphabetically.

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

TP_DATE = """Task: Answer the day-of-week question.

Example 1
Today is Monday.
9 days later is Wednesday.

Example 2
Today is Friday.
11 days later is Tuesday.

Example 3
Today is Sunday.
15 days later is Monday.

Example 4
Today is Wednesday.
20 days later is Tuesday."""

# ─────────────────────────────────────────────
# Generators
# ─────────────────────────────────────────────

NAMES  = ['Alice','Bob','Charlie','David','Emma','Jack','Kate','Luke','Mary','Peter','Amy','Ben','Carl','Eva','John','Ryan','Sam','Tom','Anna','Sarah','Mike','Lucy']
OBJS   = ['ball','key','book','coin','pen','ring','hat','bag','map','phone','watch','wallet','laptop','tablet','camera','notebook','calculator','guitar','piano','violin']
WORDS  = ['apple','banana','cherry','desk','chair','table','lion','tiger','zebra','monkey','bird','dog','cat','violin','guitar','piano','orange','kiwi','grape','window','door','computer','bottle','airplane','car','train','shoe','shirt','hat','river','cloud','stone','flame','sword','crown','tower','forest','ocean','bridge']
DAYS   = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

def mc_opts(ans, pool, n=4):
    opts = {ans}
    attempts = 0
    while len(opts) < n and attempts < 200:
        opts.add(random.choice(pool))
        attempts += 1
    opts = sorted(list(opts))[:n]
    if ans not in opts:
        opts[-1] = ans
        opts = sorted(opts)
    letter = LETTERS[opts.index(ans)]
    opts_str = "Options:\n" + "\n".join(f"{LETTERS[i]}) {opts[i]}" for i in range(len(opts)))
    return opts_str, letter

def build_task(teacher_block, task_type, q_text, opts_str, letter, task_id):
    tp = teacher_block + "\n\nQuestion\n" + q_text + "\n" + opts_str
    sp = "Task: " + teacher_block.split("\n")[0].replace("Task: ","") + "\n\nQuestion\n" + q_text + "\n" + opts_str
    return {"id": task_id, "task": task_type, "teacher_prompt": tp, "student_prompt": sp, "answer": letter}

tasks = []
tid = 1

# ── 1. tracking_shuffled_objects (2-person, 1 swap) — 10 tasks ──
objs_pairs = [('red ball','blue ball'),('key','coin'),('book','pen'),('hat','bag'),('laptop','tablet'),
              ('ring','watch'),('guitar','piano'),('camera','notebook'),('phone','wallet'),('map','mirror')]
name_pairs = [('Alice','Bob'),('Tom','Emma'),('John','Sarah'),('Mike','Lucy'),('David','Anna'),
              ('Jack','Kate'),('Sam','Ryan'),('Carl','Eva'),('Peter','Mary'),('Ben','Amy')]
q_objs_2p = [('red ball','blue ball'),('key','coin'),('book','pen'),('hat','bag'),('laptop','tablet'),
             ('ring','watch'),('guitar','piano'),('camera','notebook'),('phone','wallet'),('map','mirror')]

for i in range(10):
    n1, n2 = name_pairs[i]
    o1, o2 = q_objs_2p[i]
    # after 1 swap: n1 gets o2, n2 gets o1 — ask about o1 → answer: n2
    ask_obj = o1
    ans_name = n2
    distractor_names = [x for x in NAMES if x not in [n1, n2]][:2]
    pool = sorted([ans_name, n1] + distractor_names)
    opts_str, letter = mc_opts(ans_name, pool, n=4)
    q_text = f"{n1} has the {o1}.\n{n2} has the {o2}.\nSwap {n1} and {n2}.\nWho has the {ask_obj}?"
    tasks.append(build_task(TP_TRACKING_2P, "tracking_shuffled_objects", q_text, opts_str, letter, tid)); tid += 1

# ── 2. tracking_shuffled_objects (3-person, 2 swaps) — 10 tasks ──
triples = [
    (('John','Mary','Peter'),('laptop','tablet','camera'),(0,2),(1,2)),
    (('David','Anna','Carl'),('notebook','calculator','guitar'),(0,1),(1,2)),
    (('Jack','Kate','Luke'),('phone','watch','wallet'),(0,2),(1,2)),
    (('Alice','Bob','Charlie'),('ring','coin','key'),(0,1),(0,2)),
    (('Tom','Sam','Eva'),('hat','bag','map'),(1,2),(0,1)),
    (('Amy','Ben','Ryan'),('book','pen','hat'),(0,2),(1,2)),
    (('Sarah','Mike','Lucy'),('ball','doll','cube'),(0,1),(1,2)),
    (('Peter','Mary','Anna'),('sword','crown','shield'),(0,2),(0,1)),
    (('Carl','Eva','John'),('flame','stone','river'),(1,2),(0,2)),
    (('Kate','Luke','Jack'),('tower','forest','ocean'),(0,1),(0,2)),
]
for names3, objs3, sw1, sw2 in triples:
    n = list(names3); o = list(objs3)
    n[sw1[0]], n[sw1[1]] = n[sw1[1]], n[sw1[0]]
    o[sw1[0]], o[sw1[1]] = o[sw1[1]], o[sw1[0]]
    n[sw2[0]], n[sw2[1]] = n[sw2[1]], n[sw2[0]]
    o[sw2[0]], o[sw2[1]] = o[sw2[1]], o[sw2[0]]
    # ask about original objs3[0]
    target_obj = objs3[0]
    ans_name = n[o.index(target_obj)]
    distractor_names = [x for x in NAMES if x not in list(names3)][:1]
    pool = sorted(list(set(list(names3) + distractor_names)))[:4]
    if ans_name not in pool: pool[-1] = ans_name; pool = sorted(pool)
    opts_str, letter = mc_opts(ans_name, pool, n=4)
    q_text = (f"{names3[0]} has the {objs3[0]}.\n{names3[1]} has the {objs3[1]}.\n{names3[2]} has the {objs3[2]}.\n"
              f"Swap {names3[sw1[0]]} and {names3[sw1[1]]}.\nSwap {names3[sw2[0]]} and {names3[sw2[1]]}.\nWho has the {target_obj}?")
    tasks.append(build_task(TP_TRACKING_3P, "tracking_shuffled_objects", q_text, opts_str, letter, tid)); tid += 1

# ── 3. navigate (coordinates) — 10 tasks ──
def nav_coord():
    x, y = random.randint(0,5), random.randint(0,5)
    start = f"Start ({x},{y})"
    steps = random.randint(3,4)
    dirs = ['North','South','East','West']
    lines = [start]
    for _ in range(steps):
        d = random.choice(dirs); dist = random.randint(1,4)
        lines.append(f"{d} {dist}")
        if d=='North': y+=dist
        elif d=='South': y-=dist
        elif d=='East': x+=dist
        elif d=='West': x-=dist
    ans = f"({x},{y})"
    pool = {ans}
    while len(pool)<4:
        pool.add(f"({x+random.randint(-3,3)},{y+random.randint(-3,3)})")
    pool = sorted(list(pool))
    if ans not in pool: pool[-1]=ans; pool=sorted(pool)
    opts_str = "Options:\n"+"\n".join(f"{LETTERS[i]}) {pool[i]}" for i in range(4))
    letter = LETTERS[pool.index(ans)]
    return "\n".join(lines), opts_str, letter

for _ in range(10):
    q_text, opts_str, letter = nav_coord()
    tasks.append(build_task(TP_NAVIGATE_COORD, "navigate", q_text, opts_str, letter, tid)); tid += 1

# ── 4. navigate (directions) — 10 tasks ──
DIR_ORDER = ['North','East','South','West']
def nav_dir():
    face = random.choice(DIR_ORDER)
    turns = random.randint(2,4)
    lines = [f"Face {face}."]
    idx = DIR_ORDER.index(face)
    for _ in range(turns):
        t = random.choice(['Turn Left.','Turn Right.'])
        lines.append(t)
        if t=='Turn Left.': idx=(idx-1)%4
        else: idx=(idx+1)%4
    ans = DIR_ORDER[idx]
    pool = sorted(DIR_ORDER)
    opts_str = "Options:\n"+"\n".join(f"{LETTERS[i]}) {pool[i]}" for i in range(4))
    letter = LETTERS[pool.index(ans)]
    return "\n".join(lines), opts_str, letter

for _ in range(10):
    q_text, opts_str, letter = nav_dir()
    tasks.append(build_task(TP_NAVIGATE_DIR, "navigate", q_text, opts_str, letter, tid)); tid += 1

# ── 5. formal_fallacies — 10 tasks ──
VALID_ARGS = [
    ("All roses are flowers.\nThis is a rose.\nTherefore this is a flower.", True),
    ("All doctors studied medicine.\nSarah is a doctor.\nTherefore Sarah studied medicine.", True),
    ("All mammals breathe.\nA dog is a mammal.\nTherefore a dog breathes.", True),
    ("All students have an ID.\nJohn is a student.\nTherefore John has an ID.", True),
    ("All planets orbit a star.\nEarth is a planet.\nTherefore Earth orbits a star.", True),
]
INVALID_ARGS = [
    ("All fish swim.\nNemo swims.\nTherefore Nemo is a fish.", False),
    ("All cars have wheels.\nThis vehicle has wheels.\nTherefore this vehicle is a car.", False),
    ("All programmers know Python.\nJohn knows Python.\nTherefore John is a programmer.", False),
    ("All birds have wings.\nThis animal has wings.\nTherefore this animal is a bird.", False),
    ("All teachers work at schools.\nAlice works at a school.\nTherefore Alice is a teacher.", False),
]
ff_args = VALID_ARGS + INVALID_ARGS
random.shuffle(ff_args)
for arg_text, is_valid in ff_args[:10]:
    ans = "B" if is_valid else "A"   # A=Invalid, B=Valid
    opts_str = "Options:\nA) Invalid\nB) Valid"
    tasks.append(build_task(TP_FORMAL, "formal_fallacies", arg_text, opts_str, ans, tid)); tid += 1

# ── 6. logical_deduction — 10 tasks ──
LOGIC_CHAINS = [
    ("Every square is a rectangle.\nEvery rectangle has four sides.\nShape A is a square.\nDoes Shape A have four sides?", "B"),
    ("Every programmer writes code.\nEveryone who writes code uses a computer.\nMike is a programmer.\nDoes Mike use a computer?", "B"),
    ("Every elephant is an animal.\nEvery animal needs water.\nDumbo is an elephant.\nDoes Dumbo need water?", "B"),
    ("Every teacher works at a school.\nEveryone working at a school has an ID card.\nAlice is a teacher.\nDoes Alice have an ID card?", "B"),
    ("Every bird has wings.\nEvery animal with wings can flap.\nRobin is a bird.\nCan Robin flap?", "B"),
    ("Every engineer knows math.\nEveryone who knows math can solve equations.\nEmma is an engineer.\nCan Emma solve equations?", "B"),
    ("Every whale is a mammal.\nEvery mammal is warm-blooded.\nBlue is a whale.\nIs Blue warm-blooded?", "B"),
    ("Every chef cooks food.\nEveryone who cooks food uses a stove.\nGordon is a chef.\nDoes Gordon use a stove?", "B"),
    ("Every athlete trains daily.\nEveryone who trains daily stays fit.\nUsain is an athlete.\nDoes Usain stay fit?", "B"),
    ("Every pilot flies planes.\nEveryone who flies planes has a license.\nCaptain Ray is a pilot.\nDoes Captain Ray have a license?", "B"),
]
for q_text, letter in LOGIC_CHAINS:
    opts_str = "Options:\nA) No\nB) Yes"
    tasks.append(build_task(TP_LOGICAL, "logical_deduction", q_text, opts_str, letter, tid)); tid += 1

# ── 7. boolean_expressions — 10 tasks ──
def bool_expr():
    exprs = [
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
    ]
    return exprs

bool_exprs = bool_expr()
for expr_text, result in bool_exprs:
    ans = "B" if result else "A"
    opts_str = "Options:\nA) False\nB) True"
    tasks.append(build_task(TP_BOOLEAN, "boolean_expressions", expr_text, opts_str, ans, tid)); tid += 1

# ── 8. multistep_arithmetic — 10 tasks ──
ARITH = [
    ("((12/3)+5)*2", 18),
    ("((20/4)+7)*3", 36),
    ("((6+2)*3)-4", 20),
    ("18/(3+3)", 3),
    ("(15-5)*(8/2)", 40),
    ("7+(18/3)-2", 11),
    ("((9-3)*4)+2", 26),
    ("(100/10)+(3*4)", 22),
    ("((5+3)*(2+2))-6", 26),
    ("(50/5)-(3*2)", 4),
]
for expr_text, ans_val in ARITH:
    distractors = sorted(list(set([ans_val, ans_val+2, ans_val-2, ans_val+4])))[:4]
    if ans_val not in distractors: distractors[-1]=ans_val; distractors=sorted(distractors)
    opts_str = "Options:\n"+"\n".join(f"{LETTERS[i]}) {distractors[i]}" for i in range(4))
    letter = LETTERS[distractors.index(ans_val)]
    tasks.append(build_task(TP_ARITHMETIC, "multistep_arithmetic", expr_text, opts_str, letter, tid)); tid += 1

# ── 9. dyck_language — 10 tasks ──
def gen_balanced(n):
    if n==0: return ""
    if n==2: return "()"
    if random.random()<0.5: return "("+gen_balanced(n-2)+")"
    split = random.choice(range(2,n,2))
    return gen_balanced(split)+gen_balanced(n-split)

dyck_cases = []
for _ in range(10):
    length = random.choice([6,8,10,12])
    is_bal = random.choice([True,False])
    s = gen_balanced(length)
    if not is_bal:
        pos = random.randint(0,len(s)-1)
        l = list(s); l[pos]='(' if l[pos]==')' else ')'; s="".join(l)
    ans = "A" if is_bal else "B"
    opts_str = "Options:\nA) Balanced\nB) Unbalanced"
    tasks.append(build_task(TP_DYCK, "dyck_language", s, opts_str, ans, tid)); tid += 1

# ── 10. word_sorting — 10 tasks ──
for _ in range(10):
    w = random.sample(WORDS, random.randint(4,5))
    ans_str = " ".join(sorted(w))
    pool = [ans_str]
    for _ in range(20):
        w2=w.copy(); random.shuffle(w2); j=" ".join(w2)
        if j not in pool: pool.append(j)
    pool = sorted(pool[:4])
    if ans_str not in pool: pool[-1]=ans_str; pool=sorted(pool)
    opts_str = "Options:\n"+"\n".join(f"{LETTERS[i]}) {pool[i]}" for i in range(4))
    letter = LETTERS[pool.index(ans_str)]
    q_text = " ".join(w)
    tasks.append(build_task(TP_WORD, "word_sorting", q_text, opts_str, letter, tid)); tid += 1

print(f"Generated {len(tasks)} tasks")
print("Task type distribution:")
from collections import Counter
dist = Counter(t['task'] for t in tasks)
for k,v in dist.items():
    print(f"  {k}: {v}")

# ── Inject into notebook ──
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
                'print(f"Loaded {len(tasks)} diverse tasks across all 10 task types.")\n'
            ]
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\nInjected into notebook successfully.")

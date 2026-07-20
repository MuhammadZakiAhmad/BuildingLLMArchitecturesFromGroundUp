# Skill: /resume

Trigger: user types `/resume`

## What to do

### 1. Read ../project_history/project_details.md

If unavailable, ask user to paste it.
If file doesn't exist: tell user, suggest /start-project, stop.

### 2. Find the last PAUSED session

Check Session Index for most recent row with status PAUSED.
Read that session file from ../project_history/sessions/{ID}.md.
If multiple ACTIVE rows exist: warn user, ask which before continuing.

### 3. Generate new session ID

- Count all rows in Session Index + 1 for N
- Get current datetime or ask user
- Format: `SN-YYYY-MM-DD-HH:MM`

### 4. Create new session file

../project_history/sessions/SN-YYYY-MM-DD-HH:MM.md

# Session SN-YYYY-MM-DD-HH:MM

**Session Status**: ACTIVE
**Resumed from**: {previous session ID}
**Working on**: {item 1 from previous "Next steps"}
**Starting point**: Resumed from {previous session ID}

**What is working**: {carry over from previous session}
**What is broken**: {carry over from previous session}
**Files changed**: —
**Critical commands / config / snippets**: {carry over}
**Environment facts**: {carry over}
**Dependencies added**: —
**Assumptions (unverified)**: {carry over unresolved}
**External references**: {carry over relevant}
**Open questions / blockers**: {carry over unresolved}
**Next steps**: {carry over remaining steps}
**Lessons learned**: —
**Outcome**: —

Do NOT copy Achievements, Errors, Dead Ends, or Decisions into the session file. Those live in Project Log only.

### 5. Update project_details.md Session Index

Add new row:
| SN-YYYY-MM-DD-HH:MM | YYYY-MM-DD | ACTIVE | {working on} |

If no filesystem, output both files for user to copy.

### 6. Brief the user

---
Session {SN} open — resumed from {previous ID}

Project: {name} | Goal: {goal}

Was working on: {previous "Working on"}
First step now: {item 1 from previous "Next steps"}

Broken going in: {previous "What is broken"}
Dead ends to avoid: {recent entries from Project Log > Dead Ends}
Blockers: {previous "Open questions / blockers"}

To get set up: {previous "Critical commands / config / snippets"}

Ready.
---

## Edge cases

- No file found → suggest /start-project
- Multiple ACTIVE sessions → ask user which before proceeding
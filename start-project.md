# Skill: /start-project

Trigger: user types `/start-project`

## What to do

### 1. Interview the user

Ask in one message:

1. Project name?
2. End goal / definition of done?
3. Stack / tools / languages?
4. Greenfield or does something already exist?
5. Known constraints (deadlines, hard requirements, things to avoid)?
6. What are we working on right now in this first session?

### 2. Determine session ID

- Format: `S1-YYYY-MM-DD-HH:MM`
- Get current datetime or ask user

### 3. Create two files

If no filesystem, output both for user to copy.

---

**File 1: ../project_history/project_details.md**

# Project: {name}

## Overview
- **Goal**: {goal}
- **Stack**: {stack}
- **Started**: {datetime}
- **Project Status**: ACTIVE

## Constraints
{constraints or "None specified"}

---

## Project Log

### Achievements
<!-- Append-only. Format: [SN-YYYY-MM-DD-HH:MM] {what} — {how} — {where} -->

### All Errors Encountered
<!-- Append-only. Format: [SN-YYYY-MM-DD-HH:MM] `{error}` — where: {location} — status: {resolved/partial/open} — tried: {what} -->

### Dead Ends (do not retry)
<!-- Append-only. Format: [SN-YYYY-MM-DD-HH:MM] {what tried} — why: {reason} — retry if: {condition or "never"} -->

### Decisions Made
<!-- Append-only. Format: [SN-YYYY-MM-DD-HH:MM] {decision} — why: {reason} — rejected: {alternatives} — reversible: {yes/no} -->

### Features
<!-- Append-only. Format: [SN-YYYY-MM-DD-HH:MM] {feature name} — status: {planned/in-progress/completed} — {description} -->

### Important Files
<!-- Append-only. Format: [SN-YYYY-MM-DD-HH:MM] {file path} — {description/purpose} -->

---

## Session Index
| ID | Date | Status | Working On |
|----|------|--------|------------|
| S1-YYYY-MM-DD-HH:MM | YYYY-MM-DD | ACTIVE | {first task} |

---

**File 2: ../project_history/sessions/S1-YYYY-MM-DD-HH:MM.md**

# Session S1-YYYY-MM-DD-HH:MM

**Session Status**: ACTIVE
**Resumed from**: —
**Working on**: {first task}
**Starting point**: {current state of project}

**What is working**: N/A — session just started
**What is broken**: N/A
**Files changed**: N/A
**Critical commands / config / snippets**: N/A
**Environment facts**: N/A
**Dependencies added**: N/A
**Assumptions (unverified)**: N/A
**External references**: N/A
**Open questions / blockers**: N/A
**Next steps**: 1. {first concrete action}
**Lessons learned**: N/A
**Outcome**: —

---

### 4. Confirm

Say: "Session S1 is open. Let's get to work."
# Skill: /complete-session

Trigger: user types `/complete-session`

## What to do

### 1. Read ../project_history/project_details.md

If unavailable, ask user to paste it.
Find ACTIVE row in Session Index. Read that session file.

### 2. Extract from conversation — same depth as /pause

Same extraction as /pause. Additionally:
- **Outcome**: Yes / Partial / No — one line
- **Lessons learned**: stack, process, decisions worth remembering

### 3. Write updates

1. Append to the 4 Project Log sections in project_details.md (tag with session ID)
2. Update Session Index row: change status to COMPLETED
3. In session file: replace `**Session Status**: ACTIVE` with `**Session Status**: COMPLETED — {datetime}`
4. Fill all session fields. Set **Outcome**.

### 4. Ask: is the project itself done?

"Is the project complete, or just this session?"

**Session only**: leave `**Project Status**: ACTIVE`. Done.

**Project complete**:
- Change `**Project Status**: ACTIVE` to `**Project Status**: COMPLETED — {datetime}`
- Append to project_details.md after Session Index:

---

## Final Summary

**Completed**: {datetime}
**Total sessions**: {N}

**What was built / achieved**:
{description — reference Project Log > Achievements for full list}

**Key decisions**:
{most impactful from Project Log > Decisions Made}

**Lessons learned**:
{consolidate from all session files "Lessons learned" fields}

**Known issues / tech debt left behind**:
{item or "None"}

---

### 5. Confirm

If no filesystem, print all updated files in full for user to copy.
# Skill: /pause

Trigger: user types `/pause`

## Goal

Zero re-discovery next session. No memory, no scrollback. Write everything down.

---

## Step 1 — Read ../project_history/project_details.md

If unavailable, ask user to paste it. Find the ACTIVE row in Session Index to get current session ID.
Read ../project_history/sessions/{current-session-ID}.md for the session file.

---

## Step 2 — Extract from the full conversation

### For the Project Log (append to project_details.md — never overwrite)

**Achievements** — every completed item this session:
- What done, how done, where it lives (path/function/endpoint)
- Tag with session ID

**All errors encountered** — every error that appeared:
- Exact error text, where it occurs, status: resolved/partial/open, what was tried
- Tag with session ID

**Dead ends** — everything tried and abandoned:
- What tried, why failed, retry condition or "never"
- Tag with session ID

**Decisions made** — every decision taken:
- What decided, why, alternatives rejected, reversible yes/no
- Tag with session ID

### For the session file (update current session)

**Working on**: exact task at pause, state (not started / in progress / almost done / blocked), last action taken

**What is working**: concrete — paths, endpoints, commands, how tested

**What is broken**: symptom, exact last error message, what was tried

**Files changed**: every file created/modified/deleted, what changed

**Critical commands / config / snippets**: exact commands, env vars, config values, ports, flags, masked secrets

**Environment facts**: runtime quirks, must-run commands before resuming, gotchas

**Dependencies added**: package + version for everything added this session

**Assumptions (unverified)**: assumed true but not confirmed — risk: high/medium/low

**External references**: URLs/docs consulted, why each matters going forward

**Open questions / blockers**: unanswered questions, pending decisions — blocking: yes/no

**Next steps**: ordered exact actions. Item 1 = first 5 minutes of next session

**Lessons learned**: anything worth remembering about stack, approach, tooling

---

## Step 3 — Write updates

1. Append new entries to the 4 Project Log sections in project_details.md
2. Update Session Index row: change status to PAUSED
3. In the session file: replace `**Session Status**: ACTIVE` with `**Session Status**: PAUSED — {datetime}`
4. Fill all session fields with extracted content
5. Do not add or remove fields. Use exact field names.

---

## Step 4 — Confirm

If no filesystem, print both updated files in full for user to copy.
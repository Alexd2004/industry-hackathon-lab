# Submissions Guide - IEEE YP Industry Hackathon

**Window:** Fri Oct 2, 2026 5:00 PM MDT → Sun Oct 4, 2026 12:00 PM MDT sharp - no exceptions.
All times are **MDT (Calgary local, UTC-6)**.

## How to submit (one issue per team)

1. Go to **Issues → New Issue → Hackathon Submission → Get started** (repo goes public at 5 PM Oct 2; the form is invisible while private).
2. Fill all 9 fields: team name, 2–5 members with `@handles`, project stream, project title, tagline (3 lines max), About (Markdown + LaTeX `$...$`), **min 2 / max 5 screenshots** (10 MB max each), demo video/live-site **link** (optional but recommended - YouTube unlisted or Loom; direct upload capped at 10 MB max), additional info.
3. The validator bot comments within a minute (`needs-review`, or `needs-fix` / `late` as applicable).

## Editing rules

- Only the teammate who clicked Submit can edit the issue body; other teammates post updates as **comments** (the original author can copy them into the body).
- Drag-drop images/video into the body or comments at any time before lock.
- Broken submission? Ask an organizer to **delete** the issue (personal-repo owner can erase instantly), then resubmit within the window.

## Deadline

Submissions close **Sunday, Oct 4 @ 12:00 PM MDT sharp - no exceptions**.

## Judging ops (1–4 PM MDT)

- Labels: `submission → needs-review/under-review → judged → winner`, plus `needs-fix`, `late`, `duplicate`.
- Export: `python scripts/export_submissions.py nagusubra/industry-hackathon-lab --out submissions.csv`
  or `gh issue list --label submission --state all --limit 200 --json number,title,author,createdAt,url,labels`.
- Stage pitches, live vote, and winners 4 PM MDT.

## Finals: shortlist, stage pitches, and live vote

- Judges shortlist standout teams from each judging room.
- Shortlisted teams pitch live on stage in front of everyone on Sunday afternoon.
- Judges pick **1st, 2nd, and 3rd prizes** from the stage finalists; the audience picks **Fan Favourite ($100)** by live vote.

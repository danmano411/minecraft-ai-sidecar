# Session Export & Paxel Upload Guide

How to export a Claude Code session transcript and upload it to YC Paxel for analysis.

---

## What is Paxel?

Paxel (`paxel.ycombinator.com`) is a YC tool that analyzes Claude Code sessions — it reads your coding agent transcripts, scores them across 5 axes, and generates a report. It runs analysis locally inside a Docker container (your code never leaves your machine; only aggregate scores and metadata are uploaded).

---

## Prerequisites

- **Docker Desktop** installed and running — https://docs.docker.com/get-docker/
- **Python 3.12+** (already required for this project)
- A Claude Code session JSONL file (see Step 1)

---

## Step 1 — Find your session JSONL

Claude Code stores session transcripts as JSONL files at:

```
C:\Users\<you>\.claude\projects\<project-slug>\<session-id>.jsonl
```

The project slug is a sanitized version of your project path, e.g.:
```
c--Users-danma-Documents-Dan-Projects-Minecraft-AI-Helper
```

The session ID is a UUID like `1952e55a-72f1-4a83-bb61-0e3c4f2b2a95`.

To find yours, look for the most recently modified `.jsonl` file:
```powershell
Get-ChildItem "$env:USERPROFILE\.claude\projects" -Recurse -Filter "*.jsonl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 5 FullName, LastWriteTime
```

---

## Step 2 — Export to Markdown (optional, for human review)

Use the script below to convert a JSONL session to a readable `.md` file.
Edit the two path constants at the top before running:

```python
# export_session.py  —  place this file in your project root (it is gitignored)
python export_session.py
```

The output is written to the path set in `OUT`. Move it to `sessions/` afterward:
```powershell
Move-Item "C:\Users\<you>\Desktop\session.md" "sessions\session_name.md"
```

The `sessions/` folder is gitignored — transcripts are never committed.

---

## Step 3 — Upload to Paxel

Run this from inside your project folder (replace the path placeholder):

```bash
cd "path/to/your-project" && curl -fsSL https://paxel.ycombinator.com/upload.sh | bash
```

**First run:** Paxel will open a browser for authentication. Sign in — check spam if the login email is slow. Your token is saved to `~/.paxel/token` so subsequent runs skip this step.

**What it does:**
1. Checks Docker is available
2. Pulls the `ghcr.io/yc-software/paxel-client:latest` container (first run only)
3. Scans `~/.claude/projects/` for Claude Code session transcripts
4. Runs analysis locally in Docker (git metrics, session summary, decision extraction, scoring)
5. Uploads only aggregate metadata — file bodies stay local
6. Emails you when the report is ready

**Re-runs are incremental** — LLM results are cached locally so only new session activity is re-analyzed.

Results: **https://paxel.ycombinator.com/reports**

---

## Export Script

Save this as `export_session.py` in your project root. Edit `JSONL` and `OUT` for your paths.

```python
"""
One-shot script: converts a Claude Code JSONL session transcript to a clean Markdown file.
Edit JSONL and OUT below, then run: python export_session.py
"""

import json
from pathlib import Path

JSONL = Path(r"C:\Users\<you>\.claude\projects\<project-slug>\<session-id>.jsonl")
OUT   = Path(r"C:\Users\<you>\Desktop\session_export.md")


def extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            t = block.get("text", "").strip()
            if t:
                parts.append(t)
    return "\n\n".join(parts)


def tool_call_summary(content) -> list[str]:
    lines = []
    if not isinstance(content, list):
        return lines
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            name = block.get("name", "?")
            inp  = block.get("input", {})
            detail = ""
            if "file_path" in inp:
                detail = inp["file_path"]
            elif "pattern" in inp:
                detail = inp["pattern"]
            elif "command" in inp:
                cmd = inp["command"]
                detail = cmd[:80] + ("..." if len(cmd) > 80 else "")
            elif "description" in inp:
                detail = inp["description"][:80]
            elif "old_string" in inp:
                detail = "(edit)"
            lines.append(f"*`[tool: {name}{(' — ' + detail) if detail else ''}]`*")
    return lines


def build_markdown(jsonl_path: Path) -> str:
    records = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    seen_uuids: set[str] = set()
    deduped = []
    for r in records:
        uid = r.get("uuid")
        if uid and uid in seen_uuids:
            continue
        if uid:
            seen_uuids.add(uid)
        deduped.append(r)

    md_parts = [
        "# Claude Code Session Transcript",
        "",
        "---",
        "",
    ]

    msg_count = 0
    for r in deduped:
        rtype = r.get("type")
        msg   = r.get("message", {})
        role  = msg.get("role", "")
        content = msg.get("content", "")

        if rtype not in ("user", "assistant"):
            continue

        if role == "user" and isinstance(content, list):
            if all(b.get("type") == "tool_result" for b in content if isinstance(b, dict)):
                continue

        if role == "user":
            text = extract_text(content)
            if not text:
                continue
            msg_count += 1
            md_parts += [f"## User", "", text, "", "---", ""]

        elif role == "assistant":
            text  = extract_text(content)
            tools = tool_call_summary(content) if isinstance(content, list) else []
            if not text and not tools:
                continue
            msg_count += 1
            md_parts += [f"## Claude", ""]
            if text:
                md_parts += [text, ""]
            if tools:
                md_parts += tools + [""]
            md_parts += ["---", ""]

    md_parts.insert(4, f"*{msg_count} conversation turns*\n")
    return "\n".join(md_parts)


if __name__ == "__main__":
    print("Converting...")
    md = build_markdown(JSONL)
    OUT.write_text(md, encoding="utf-8")
    print(f"Written to: {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")
```

---

## Notes for Claude instances

- `export_session.py` is listed in `.gitignore` — do not commit it
- The `sessions/` folder is also gitignored — transcripts are never committed
- Paxel authentication token lives at `~/.paxel/token` — no need to re-authenticate on subsequent runs
- To find the active session JSONL, check the task output paths used during the session — they contain the session UUID (e.g. `...tasks/<session-id>/...`)

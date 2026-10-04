"""Safety scan of the recorded run before it is committed (the repo is public; Crew Plan step 17).

Checks results/rounds_recorded.csv and results/decisions_recorded.jsonl for:
  secret       an API key or credential header
  test_id      an account id from the frozen test set (results/split.json)
  label_key    a JSON key or csv column that carries a label or a forbidden field (LABEL_KEYS)
  forbidden    a forbidden column name used as a word in any text (age, job, gender, ...)
Run: python -m softsignal.recorded_check   (exit 1 and a list of problems if anything is found)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from softsignal.agents.contracts import LABEL_KEYS, frozen_test_ids
from softsignal.crew import DECISIONS_RECORDED, ROUNDS_RECORDED
from softsignal.features import FORBIDDEN

SECRET = re.compile(r"sk-[A-Za-z0-9_-]{16,}|ANTHROPIC_[A-Z_]*=|x-api-key|api[_-]?key\s*[:=]|bearer\s+[A-Za-z0-9._-]{16,}",
                    re.IGNORECASE)
FORBIDDEN_WORD = re.compile(r"\b(" + "|".join(sorted(map(re.escape, FORBIDDEN))) + r")\b", re.IGNORECASE)
ID_TOKEN = re.compile(r"\b[A-Za-z]\d{5,}\b")


def _keys(node) -> list[str]:
    if isinstance(node, dict):
        return [str(k) for k in node] + [k for v in node.values() for k in _keys(v)]
    if isinstance(node, list):
        return [k for v in node for k in _keys(v)]
    return []


def scan_text(text: str, test_ids: frozenset, where: str) -> list[str]:
    """Problems in one file's text: secret, test_id and forbidden words (label keys are checked by scan_file)."""
    out = [f"{where}: secret: {m.group()[:12]}..." for m in SECRET.finditer(text)]
    out += [f"{where}: test_id: {t}" for t in sorted(set(ID_TOKEN.findall(text)) & test_ids)]
    out += [f"{where}: forbidden: {w}" for w in sorted({m.group().lower() for m in FORBIDDEN_WORD.finditer(text)})]
    return out


def scan_file(path: Path, test_ids: frozenset) -> list[str]:
    """All problems in a recorded rounds csv or decisions jsonl. A missing file is a problem."""
    path = Path(path)
    if not path.exists():
        return [f"{path.name}: missing"]
    text = path.read_text(encoding="utf-8")
    body = text
    keys: list[str] = []
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            try:
                keys += _keys(json.loads(line))
            except json.JSONDecodeError:
                return [f"{path.name}: not valid JSON lines"]
        # key names are checked below; the word scan looks at values only
        body = "\n".join(json.dumps(_values(json.loads(line))) for line in text.splitlines())
    else:
        header, _, body = text.partition("\n")
        keys = header.split(",")
    out = scan_text(body, test_ids, path.name)
    out += [f"{path.name}: label_key: {k}" for k in sorted(set(keys) & LABEL_KEYS)]
    return out


def _values(node):
    if isinstance(node, dict):
        return [_values(v) for v in node.values()]
    if isinstance(node, list):
        return [_values(v) for v in node]
    return node


def scan_recorded(rounds: Path = ROUNDS_RECORDED, decisions: Path = DECISIONS_RECORDED,
                  test_ids: frozenset | None = None) -> list[str]:
    ids = frozen_test_ids() if test_ids is None else test_ids
    return scan_file(rounds, ids) + scan_file(decisions, ids)


def main() -> None:
    problems = scan_recorded()
    for p in problems:
        print(p)
    print("recorded files clean" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

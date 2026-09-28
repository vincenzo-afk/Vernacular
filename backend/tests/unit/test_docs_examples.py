"""
Docs-vs-code drift guard.

ARCHITECTURE.md once shipped a flagship StyleMetadata example with
`"formality": "informal"`, which is not a valid enum value and would
have failed real validation. Nothing caught it because docs are not
executed. This test extracts every ```json block in the markdown docs
that looks like a StyleMetadata payload and validates it against the
real pydantic model, so an example that cannot actually be produced or
consumed by the system fails CI instead of misleading the next reader
(human or agent).
"""

import json
import re
from pathlib import Path

import pytest

from app.schemas.style_metadata import StyleMetadata

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_FILES = [
    *REPO_ROOT.glob("*.md"),
    *(REPO_ROOT / "docs").glob("*.md"),
]

JSON_BLOCK = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)
STYLE_KEYS = {"tone", "pace", "formality", "emotion", "sarcasm_score"}


def _style_examples() -> list[tuple[str, dict]]:
    found = []
    for path in DOC_FILES:
        text = path.read_text(encoding="utf-8")
        for i, match in enumerate(JSON_BLOCK.finditer(text)):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and STYLE_KEYS <= payload.keys():
                found.append((f"{path.name}#{i}", payload))
    return found


def test_docs_contain_style_examples():
    # Guards against the regex silently matching nothing (e.g. docs
    # moved), which would make the parametrized test below vacuous.
    assert len(_style_examples()) >= 4


@pytest.mark.parametrize(
    "payload", [p for _, p in _style_examples()], ids=[n for n, _ in _style_examples()]
)
def test_doc_style_example_validates_against_real_schema(payload):
    StyleMetadata(**payload)

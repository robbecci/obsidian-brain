from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Note:
    """Rappresenta una singola nota Markdown della vault, già parsata."""

    path: Path
    relative_path: str
    title: str
    content: str
    frontmatter: dict = field(default_factory=dict)
    wikilinks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created: datetime | None = None
    modified: datetime | None = None
    word_count: int = 0

    def __repr__(self) -> str:
        return (
            f"Note(title={self.title!r}, "
            f"links={len(self.wikilinks)}, tags={len(self.tags)}, "
            f"words={self.word_count})"
        )
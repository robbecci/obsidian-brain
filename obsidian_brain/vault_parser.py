from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import frontmatter

from obsidian_brain.models import Note

DEFAULT_IGNORED_DIRS = {".obsidian", ".trash", ".git"}

WIKILINK_PATTERN = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
TAG_PATTERN = re.compile(r"(?<!\S)#([A-Za-z0-9_/-]+)")


class VaultParser:
    """Esegue il parsing di una vault Obsidian in una lista di oggetti Note."""

    def __init__(self, vault_path: str | Path, ignored_dirs: set[str] | None = None):
        self.vault_path = Path(vault_path).expanduser().resolve()
        if not self.vault_path.exists():
            raise FileNotFoundError(f"Vault non trovata: {self.vault_path}")
        if not self.vault_path.is_dir():
            raise NotADirectoryError(f"Il percorso non è una cartella: {self.vault_path}")
        self.ignored_dirs = ignored_dirs or DEFAULT_IGNORED_DIRS

    def find_markdown_files(self) -> list[Path]:
        files = []
        for md_file in self.vault_path.rglob("*.md"):
            if any(part in self.ignored_dirs for part in md_file.parts):
                continue
            files.append(md_file)
        return sorted(files)

    def parse_file(self, file_path: Path) -> Note:
        raw = file_path.read_text(encoding="utf-8-sig")
        post = frontmatter.loads(raw)

        content = post.content.strip()
        fm = dict(post.metadata)
        title = fm.get("title") or file_path.stem

        wikilinks = sorted(set(WIKILINK_PATTERN.findall(content)))

        tags_from_body = set(TAG_PATTERN.findall(content))
        tags_from_frontmatter = set()
        fm_tags = fm.get("tags")
        if fm_tags:
            if isinstance(fm_tags, str):
                tags_from_frontmatter = {fm_tags}
            elif isinstance(fm_tags, list):
                tags_from_frontmatter = {str(t) for t in fm_tags}
        tags = sorted(tags_from_body | tags_from_frontmatter)

        stat = file_path.stat()
        relative_path = file_path.relative_to(self.vault_path).as_posix()

        return Note(
            path=file_path,
            relative_path=relative_path,
            title=title,
            content=content,
            frontmatter=fm,
            wikilinks=wikilinks,
            tags=tags,
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime),
            word_count=len(content.split()),
        )

    def parse_vault(self) -> list[Note]:
        notes = []
        for md_file in self.find_markdown_files():
            try:
                notes.append(self.parse_file(md_file))
            except Exception as e:
                print(f"[WARN] Impossibile parsare {md_file}: {e}")
        return notes


def load_vault(vault_path: str | Path) -> list[Note]:
    parser = VaultParser(vault_path)
    return parser.parse_vault()
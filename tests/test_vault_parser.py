from pathlib import Path

import pytest

from obsidian_brain.vault_parser import VaultParser

SAMPLE_VAULT = Path(__file__).parent.parent / "sample_vault"


@pytest.fixture
def parser():
    return VaultParser(SAMPLE_VAULT)


def test_finds_all_markdown_files_excluding_obsidian_dir(parser):
    files = parser.find_markdown_files()
    names = {f.name for f in files}
    assert names == {"Python.md", "Data Science.md", "FastAPI.md", "Obsidian Brain.md"}
    assert not any(".obsidian" in f.parts for f in files)


def test_parses_frontmatter_fields(parser):
    notes = {n.relative_path: n for n in parser.parse_vault()}
    fastapi_note = notes["Progetti/FastAPI.md"]
    assert fastapi_note.tags == ["api", "framework"]
    assert "created" in fastapi_note.frontmatter


def test_title_falls_back_to_filename_without_frontmatter_title(parser):
    notes = {n.relative_path: n for n in parser.parse_vault()}
    ds_note = notes["Data Science.md"]
    assert ds_note.title == "Data Science"


def test_title_uses_frontmatter_when_present(parser):
    notes = {n.relative_path: n for n in parser.parse_vault()}
    ob_note = notes["Progetti/Obsidian Brain.md"]
    assert ob_note.title == "Obsidian Brain - Note di Progetto"


def test_extracts_wikilinks_including_aliased(parser):
    notes = {n.relative_path: n for n in parser.parse_vault()}
    python_note = notes["Python.md"]
    assert "Data Science" in python_note.wikilinks
    assert "FastAPI" in python_note.wikilinks


def test_extracts_tags_from_body_and_frontmatter(parser):
    notes = {n.relative_path: n for n in parser.parse_vault()}
    python_note = notes["Python.md"]
    assert "linguaggi" in python_note.tags
    assert "programmazione" in python_note.tags
    assert "linguaggio-interpretato" in python_note.tags


def test_content_excludes_frontmatter_block(parser):
    notes = {n.relative_path: n for n in parser.parse_vault()}
    python_note = notes["Python.md"]
    assert "title: Python" not in python_note.content
    assert "Python è un linguaggio interpretato" in python_note.content


def test_word_count_is_positive(parser):
    notes = parser.parse_vault()
    assert all(n.word_count > 0 for n in notes)


def test_relative_path_is_relative_to_vault_root(parser):
    notes = parser.parse_vault()
    for n in notes:
        assert not n.relative_path.startswith("/")
        assert str(SAMPLE_VAULT) not in n.relative_path
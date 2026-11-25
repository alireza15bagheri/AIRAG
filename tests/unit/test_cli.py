"""
Unit tests for the ingestion CLI (step 2.5.2 equivalent).

Exercises argument parsing, all three subcommands, human/JSON output,
file output and exit-code behaviour using only tmp_path fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.airag.ingestion.cli import create_parser, main


@pytest.fixture
def cli_tree(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# Head\n\nbody\n", encoding="utf-8")
    return tmp_path


class TestParser:
    def test_directory_defaults(self) -> None:
        args = create_parser().parse_args(["directory", "docs"])
        assert args.path == Path("docs")
        assert args.recursive is True
        assert args.fail_fast is False
        assert args.no_deduplicate is False

    def test_directory_flags(self) -> None:
        args = create_parser().parse_args(
            ["directory", "docs", "--no-recursive", "-e", ".txt", ".md",
             "--fail-fast", "--no-deduplicate", "-v", "-o", "out.json"]
        )
        assert args.recursive is False
        assert args.extensions == [".txt", ".md"]
        assert args.fail_fast is True
        assert args.no_deduplicate is True
        assert args.verbose is True
        assert args.output == Path("out.json")

    def test_files_command(self) -> None:
        args = create_parser().parse_args(["files", "a.txt", "b.md"])
        assert args.paths == [Path("a.txt"), Path("b.md")]

    def test_missing_subcommand_exits(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2


class TestCommands:
    def test_extensions_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["extensions"]) == 0
        out = capsys.readouterr().out
        assert "Supported extensions:" in out
        for ext in (".pdf", ".docx", ".md", ".txt"):
            assert ext in out

    def test_directory_human_output(
        self, cli_tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["directory", str(cli_tree)]) == 0
        out = capsys.readouterr().out
        assert "Ingestion Complete" in out
        assert "Loaded:    2" in out
        assert "Failed:    0" in out

    def test_directory_json_output(self, cli_tree: Path) -> None:
        out_path = cli_tree / "out.json"
        assert main(["directory", str(cli_tree), "-o", str(out_path)]) == 0
        payload = json.loads(out_path.read_text(encoding="utf-8"))

        assert payload["loaded"] == 2
        assert payload["failed"] == 0
        assert len(payload["results"]) == 2
        assert all(len(r["hash"]) == 64 for r in payload["results"])

    def test_directory_fail_fast_returns_error_code(
        self, cli_tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (cli_tree / "bad.txt").write_bytes(b"\x00")
        assert main(["directory", str(cli_tree), "--fail-fast"]) == 1
        assert "Error:" in capsys.readouterr().err

    def test_directory_verbose_json(
        self, cli_tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["directory", str(cli_tree), "-v"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["loaded"] == 2
        assert all(r["hash"].endswith("...") for r in payload["results"])

    def test_files_command(
        self, cli_tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["files", str(cli_tree / "a.txt")]) == 0
        out = capsys.readouterr().out
        assert "Loaded:    1" in out

    def test_files_reports_failure(
        self, cli_tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = cli_tree / "bad.txt"
        bad.write_bytes(b"\x00")
        assert main(["files", str(bad)]) == 1
        assert "Failed files:" in capsys.readouterr().out

    def test_files_verbose_and_json_output(
        self, cli_tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out_path = cli_tree / "files.json"
        assert main(["files", str(cli_tree / "a.txt"), "-v", "-o", str(out_path)]) == 0
        stdout = capsys.readouterr().out
        assert '"loaded": 1' in stdout
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["loaded"] == 1
        assert len(payload["results"][0]["hash"]) == 64

    def test_files_fail_fast_returns_error_code(
        self, cli_tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = cli_tree / "bad.txt"
        bad.write_bytes(b"\x00")
        assert main(["files", str(bad), "--fail-fast"]) == 1
        assert "Error:" in capsys.readouterr().err

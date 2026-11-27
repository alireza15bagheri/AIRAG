"""
Unit tests for the ingestion CLI (step 2.5.2 equivalent).

Exercises argument parsing, all three subcommands, human/JSON output,
file output and exit-code behaviour using only tmp_path fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.airag.ingestion.chunking.base import Chunk
from src.airag.ingestion.cli import create_parser, main
from src.airag.ingestion.pipeline import PipelineResult


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


class FakeChunker:
    def chunk(self, document: Any) -> list[Chunk]:
        return [
            Chunk(
                text=document.text,
                metadata=dict(document.metadata),
                chunk_index=0,
                end_char=len(document.text),
            )
        ]


class FakeEmbeddingPipeline:
    def run(self, chunks: list[Chunk]) -> PipelineResult:
        return PipelineResult(processed=len(chunks), written=len(chunks))


def fake_components(config: Any) -> tuple[FakeChunker, FakeEmbeddingPipeline]:
    return FakeChunker(), FakeEmbeddingPipeline()


class TestIngestCommand:
    def test_parser_defaults(self) -> None:
        args = create_parser().parse_args(["ingest", "docs"])
        assert args.path == Path("docs")
        assert args.config is None
        assert args.chunker is None
        assert args.chunk_size is None
        assert args.overlap is None
        assert args.no_recursive is False

    def test_human_output(
        self,
        cli_tree: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("src.airag.ingestion.cli.create_components", fake_components)

        assert main(["ingest", str(cli_tree)]) == 0
        out = capsys.readouterr().out
        assert "Pipeline Complete" in out
        assert "Succeeded: 2" in out
        assert "Written:   2" in out

    def test_verbose_json_output(
        self, cli_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.airag.ingestion.cli.create_components", fake_components)
        out_path = cli_tree / "pipeline.json"

        assert main(["ingest", str(cli_tree), "-v", "-o", str(out_path)]) == 0
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["files"] == 2
        assert payload["written"] == 2
        assert len(payload["results"]) == 2

    def test_config_and_cli_overrides(
        self, cli_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = cli_tree / "ingestion.yaml"
        config_path.write_text(
            "chunker: semantic\nchunk_size: 120\noverlap: 12\nrecursive: true\n",
            encoding="utf-8",
        )
        captured: dict[str, Any] = {}

        def capture(config: Any) -> tuple[FakeChunker, FakeEmbeddingPipeline]:
            captured["chunker"] = config.chunker
            captured["chunk_size"] = config.chunk_size
            captured["overlap"] = config.overlap
            captured["recursive"] = config.recursive
            return fake_components(config)

        monkeypatch.setattr("src.airag.ingestion.cli.create_components", capture)

        assert main([
            "ingest", str(cli_tree), "--config", str(config_path),
            "--chunker", "fixed", "--chunk-size", "200", "--overlap", "5",
            "--no-recursive",
        ]) == 0
        assert captured == {
            "chunker": "fixed",
            "chunk_size": 200,
            "overlap": 5,
            "recursive": False,
        }

    def test_invalid_config_path_returns_error(
        self, cli_tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([
            "ingest", str(cli_tree), "--config", str(cli_tree / "missing.yaml")
        ]) == 1
        assert "Error:" in capsys.readouterr().err

    def test_no_success_returns_error_code(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("src.airag.ingestion.cli.create_components", fake_components)
        assert main(["ingest", str(tmp_path)]) == 1
        assert "Succeeded: 0" in capsys.readouterr().out

    def test_failed_file_reported(
        self, cli_tree: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (cli_tree / "bad.txt").write_bytes(b"\x00")
        monkeypatch.setattr("src.airag.ingestion.cli.create_components", fake_components)

        assert main(["ingest", str(cli_tree)]) == 1
        out = capsys.readouterr().out
        assert "Failed:    1" in out
        assert "Failed files:" in out

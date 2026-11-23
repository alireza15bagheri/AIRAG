"""
Markdown Document Loader.

Step 2.1.3: MarkdownLoader implementation with YAML frontmatter parsing,
heading extraction, and verbatim preservation of code blocks and tables.
Step 2.2.4: Markdown metadata extraction (frontmatter, heading hierarchy).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from src.airag.ingestion.exceptions import LoadError
from src.airag.ingestion.loaders.base import (
    DocumentLoader,
    ExtractedDocument,
    LoadResult,
)

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(?P<block>.*?)\r?\n---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL,
)


def _load_yaml(block: str) -> Any:
    """
    Parse a YAML block with PyYAML's safe loader.

    Some distributions ship a ruamel.yaml-backed ``yaml`` shim where
    ``safe_load`` has been removed; fall back to the equivalent
    ``YAML(typ="safe")`` API in that case.

    Args:
        block: YAML text to parse.

    Returns:
        The parsed Python object (usually a dict for frontmatter).
    """
    import yaml

    try:
        return yaml.safe_load(block)
    except AttributeError:
        yaml_factory = getattr(yaml, "YAML", None)
        if yaml_factory is None:
            raise AttributeError("No safe YAML loader available")
        return yaml_factory(typ="safe", pure=True).load(block)


_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


class _ParsedMarkdown(NamedTuple):
    """Internal result of parsing a markdown file."""

    body: str
    frontmatter: dict[str, Any]
    headings: list[dict[str, Any]]
    table_count: int
    has_code_blocks: bool


class MarkdownLoader(DocumentLoader):
    """
    Markdown document loader for .md and .markdown files.

    Parses YAML frontmatter, records the heading hierarchy, counts pipe
    tables, and detects fenced/indented code blocks. The body text is returned
    verbatim (after frontmatter removal) so downstream chunkers see the
    original markdown source.

    Supported extensions: .md, .markdown (case-insensitive)
    """

    SUPPORTED_EXTENSIONS = frozenset({".md", ".markdown"})

    def load(self, file_path: Path) -> LoadResult:
        """
        Load and extract text from a markdown document.

        Args:
            file_path: Path to the markdown file (str or Path).

        Returns:
            LoadResult containing the extracted document or error information.

        Raises:
            LoadError: Unsupported extension or a file that cannot be decoded.
        """
        file_path = Path(file_path)
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise LoadError(
                f"Unsupported file extension: {file_path.suffix}. "
                "Expected .md or .markdown",
                source_path=str(file_path),
            )

        try:
            # Read once, tolerate a UTF-8 BOM.
            raw_text = file_path.read_text(encoding="utf-8-sig")
            parsed = self._parse(raw_text, file_path)
            metadata = self._build_metadata(file_path, parsed)

            logger.info(
                "Successfully loaded Markdown",
                extra={
                    "source_path": str(file_path),
                    "char_count": len(parsed.body),
                    "heading_count": len(parsed.headings),
                    "table_count": parsed.table_count,
                },
            )

            # Non-paginated format: pages holds the single full-text "page"
            # (page_count=1), consistent with PDFLoader's pages contract.
            extracted_doc = ExtractedDocument(
                text=parsed.body,
                metadata=metadata,
                pages=[parsed.body],
            )

            return LoadResult(
                success=True,
                document=extracted_doc,
                error=None,
                source_path=file_path,
            )

        except LoadError:
            raise
        except Exception as e:
            raise LoadError(
                f"Failed to load Markdown: {e}",
                source_path=str(file_path),
                cause=e,
            ) from e

    def _parse(self, raw_text: str, file_path: Path) -> _ParsedMarkdown:
        """
        Split frontmatter from the body and scan the body structure.

        Args:
            raw_text: Full text of the markdown file.
            file_path: Source path, used for structured log context.

        Returns:
            Parsed markdown components. Malformed frontmatter is preserved
            verbatim at the top of the body and logged as a warning.
        """
        frontmatter: dict[str, Any] = {}
        body = raw_text

        match = _FRONTMATTER_RE.match(raw_text)
        if match is not None:
            block = match.group("block")
            parsed_yaml: Any = None
            try:
                parsed_yaml = _load_yaml(block)
            except yaml.YAMLError as e:
                logger.warning(
                    "Malformed YAML frontmatter; preserving it as body text",
                    extra={"source_path": str(file_path), "error": str(e)},
                )
            if isinstance(parsed_yaml, dict):
                frontmatter = self._json_sanitize(parsed_yaml)
                body = raw_text[match.end() :]
            elif parsed_yaml is not None:
                logger.warning(
                    "YAML frontmatter is not a mapping; preserving it as body text",
                    extra={"source_path": str(file_path)},
                )

        headings: list[dict[str, Any]] = []
        table_count = 0
        has_code_blocks = False
        fence_char: str | None = None
        fence_len = 0
        previous_line = ""

        for line in body.splitlines():
            fence = _FENCE_RE.match(line)

            if fence_char is not None:
                # Inside a fenced block: content is verbatim; only a matching
                # closing fence (same char, at least as long) ends the block.
                if (
                    fence is not None
                    and fence.group(1)[0] == fence_char
                    and len(fence.group(1)) >= fence_len
                ):
                    fence_char = None
                previous_line = line
                continue

            if fence is not None:
                fence_char = fence.group(1)[0]
                fence_len = len(fence.group(1))
                has_code_blocks = True
                previous_line = line
                continue

            if line.startswith(("    ", "\t")):
                has_code_blocks = True

            # Skip escaped headings (e.g., \# not a heading)
            if line.startswith("\\#"):
                previous_line = line
                continue

            heading = _HEADING_RE.match(line)
            if heading is not None:
                heading_text = heading.group(2).strip()
                if heading_text:  # Skip empty/whitespace-only headings
                    headings.append(
                        {
                            "level": len(heading.group(1)),
                            "text": heading_text,
                        }
                    )

            if (
                "|" in line
                and "|" in previous_line
                and _TABLE_SEPARATOR_RE.match(line) is not None
            ):
                table_count += 1

            previous_line = line

        return _ParsedMarkdown(
            body=body,
            frontmatter=frontmatter,
            headings=headings,
            table_count=table_count,
            has_code_blocks=has_code_blocks,
        )

    @staticmethod
    def _json_sanitize(obj: Any) -> Any:
        """
        Recursively convert non-JSON-serializable values to JSON-safe equivalents.

        - datetime.date / datetime.datetime -> ISO-8601 string
        - Other non-serializable types -> str() fallback
        """
        if isinstance(obj, dict):
            return {k: MarkdownLoader._json_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MarkdownLoader._json_sanitize(v) for v in obj]
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        # Fallback for any other non-serializable types (e.g., yaml tags)
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    def _build_metadata(
        self, file_path: Path, parsed: _ParsedMarkdown
    ) -> dict[str, Any]:
        """
        Build metadata for a markdown document.

        Populates the following fields (per step 2.2.4):
            - source: original file path
            - page_count: 1 (non-paginated format)
            - format: "markdown"
            - frontmatter: parsed YAML frontmatter as dict, or {} if absent/invalid
            - headings: list of {"level": int, "text": str} in document order
            - heading_count: len(headings)
            - table_count: number of pipe tables detected
            - has_code_blocks: bool, true if any fenced/indented code blocks found

        Args:
            file_path: Path to the source file.
            parsed: Parsed markdown components.

        Returns:
            Dictionary containing document metadata.
        """
        metadata = {
            "source": str(file_path),
            "page_count": 1,
            "format": "markdown",
            "frontmatter": parsed.frontmatter,
            "headings": parsed.headings,
            "heading_count": len(parsed.headings),
            "table_count": parsed.table_count,
            "has_code_blocks": parsed.has_code_blocks,
        }

        logger.debug(
            "Extracted Markdown metadata",
            extra={
                "source_path": str(file_path),
                "heading_count": len(parsed.headings),
                "frontmatter_keys": len(parsed.frontmatter),
                "table_count": parsed.table_count,
                "has_code_blocks": parsed.has_code_blocks,
            },
        )

        if not parsed.frontmatter and not parsed.headings:
            logger.warning(
                "Markdown has no frontmatter or headings; metadata fields will be empty/default",
                extra={"source_path": str(file_path)},
            )

        return metadata

    def extract_metadata(self, file_path: Path) -> dict[str, Any]:
        """
        Extract metadata from a markdown file without returning its content.

        Args:
            file_path: Path to the markdown file (str or Path).

        Returns:
            Dictionary containing document metadata.

        Raises:
            LoadError: If the file cannot be read or decoded.
        """
        file_path = Path(file_path)
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise LoadError(
                f"Unsupported file extension: {file_path.suffix}. "
                "Expected .md or .markdown",
                source_path=str(file_path),
            )

        try:
            raw_text = file_path.read_text(encoding="utf-8-sig")
            parsed = self._parse(raw_text, file_path)
            return self._build_metadata(file_path, parsed)
        except LoadError:
            raise
        except Exception as e:
            raise LoadError(
                f"Failed to extract metadata from Markdown: {e}",
                source_path=str(file_path),
                cause=e,
            ) from e
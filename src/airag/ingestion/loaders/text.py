"""
Plain Text Document Loader.

Step 2.1.3: TextLoader implementation with charset-normalizer (chardet
fallback) encoding detection, line-ending normalization, and degraded UTF-8
handling for low-confidence detections.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.airag.ingestion.exceptions import LoadError
from src.airag.ingestion.loaders.base import (
    DocumentLoader,
    ExtractedDocument,
    LoadResult,
)

logger = logging.getLogger(__name__)


def _detection_confidence(match: Any) -> float:
    """
    Read a normalized 0..1 confidence score from a charset-normalizer match.

    charset-normalizer < 3.5 exposes ``confidence``; 3.5+ exposes
    ``percent_chaos`` (lower is better) instead.

    Args:
        match: A charset-normalizer CharsetMatch-like object.

    Returns:
        Confidence in the range 0..1, defaulting to 1.0 when unavailable.
    """
    confidence = getattr(match, "confidence", None)
    if confidence is not None:
        return float(confidence)
    chaos = getattr(match, "percent_chaos", None)
    if chaos is not None:
        return max(0.0, 1.0 - float(chaos) / 100.0)
    return 1.0


class TextLoader(DocumentLoader):
    """
    Plain text loader for .txt and .text files.

    Detects the encoding with charset-normalizer and falls back to chardet and
    then to UTF-8 with replacement characters. Normalizes CRLF/CR line endings
    to LF. An empty file is valid input; binary content containing NUL bytes
    is rejected.

    Supported extensions: .txt, .text (case-insensitive)
    """

    SUPPORTED_EXTENSIONS = frozenset({".txt", ".text"})
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    def load(self, file_path: Path) -> LoadResult:
        """
        Load and extract text from a plain text document.

        Args:
            file_path: Path to the text file (str or Path).

        Returns:
            LoadResult containing the extracted document or error information.

        Raises:
            LoadError: Unsupported extension, unreadable path, or binary
                content that cannot be decoded as text.
        """
        file_path = Path(file_path)
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise LoadError(
                f"Unsupported file extension: {file_path.suffix}. "
                "Expected .txt or .text",
                source_path=str(file_path),
            )

        try:
            # Read the file exactly once; detection works on these bytes.
            data = file_path.read_bytes()
            text, encoding, method = self._decode(data, file_path)
            metadata = self._build_metadata(file_path, text, encoding, method)

            logger.info(
                "Successfully loaded Text",
                extra={
                    "source_path": str(file_path),
                    "char_count": len(text),
                    "line_count": metadata["line_count"],
                    "encoding": encoding,
                    "detection_method": method,
                },
            )

            # Non-paginated format: pages holds the single full-text "page"
            # (page_count=1), consistent with PDFLoader's pages contract.
            extracted_doc = ExtractedDocument(
                text=text,
                metadata=metadata,
                pages=[text],
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
                f"Failed to load Text: {e}",
                source_path=str(file_path),
                cause=e,
            ) from e

    def _decode(self, data: bytes, file_path: Path) -> tuple[str, str, str]:
        """
        Detect the encoding and decode bytes into normalized text.

        Args:
            data: Raw file bytes.
            file_path: Source path, used for structured log context.

        Returns:
            Tuple of (normalized text, encoding name, detection method).

        Raises:
            LoadError: Decoded text contains NUL bytes (binary content).
        """
        threshold = float(
            self.get_config("confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD)
        )

        if not data:
            return "", "utf-8", "empty-file"

        encoding: str | None = None
        method = "utf-8-fallback"

        detector: Any = None
        try:
            from charset_normalizer import from_bytes as detector
        except ImportError:
            detector = None

        if detector is not None:
            try:
                match = detector(data).best()
                if (
                    match is not None
                    and match.encoding
                    and _detection_confidence(match) >= threshold
                ):
                    encoding = match.encoding
                    method = "charset-normalizer"
            except Exception as e:
                logger.warning(
                    "Encoding detection failed; falling back to UTF-8",
                    extra={"source_path": str(file_path), "error": str(e)},
                )

        if encoding is None:
            chardet_module: Any = None
            try:
                import chardet

                chardet_module = chardet
            except ImportError:
                chardet_module = None
            if chardet_module is not None:
                try:
                    detected = chardet_module.detect(data)
                    if (
                        detected.get("encoding")
                        and (detected.get("confidence") or 0.0) >= threshold
                    ):
                        encoding = detected["encoding"]
                        method = "chardet"
                except Exception as e:
                    logger.warning(
                        "chardet detection failed; falling back to UTF-8",
                        extra={"source_path": str(file_path), "error": str(e)},
                    )

        if encoding is not None:
            try:
                text = data.decode(encoding)
            except (UnicodeDecodeError, LookupError) as e:
                logger.warning(
                    "Detected encoding could not decode the file; "
                    "falling back to UTF-8 with replacement characters",
                    extra={
                        "source_path": str(file_path),
                        "encoding": encoding,
                        "error": str(e),
                    },
                )
                text = data.decode("utf-8", errors="replace")
                method = "utf-8-fallback"
                encoding = "utf-8"
        else:
            logger.warning(
                "Low-confidence encoding detection; using UTF-8 with "
                "replacement characters",
                extra={"source_path": str(file_path)},
            )
            text = data.decode("utf-8", errors="replace")
            encoding = "utf-8"
            method = "utf-8-fallback"

        if "\x00" in text:
            raise LoadError(
                "File appears to contain binary data (NUL bytes)",
                source_path=str(file_path),
            )

        return text.replace("\r\n", "\n").replace("\r", "\n"), encoding, method

    def _build_metadata(
        self, file_path: Path, text: str, encoding: str, method: str
    ) -> dict[str, Any]:
        """
        Build best-effort metadata for a text document.

        TODO(2.2.5): Expand into the full encoding-reporting spec; this stub
        records the detected encoding/method and basic text statistics.

        Args:
            file_path: Path to the source file.
            text: Decoded and normalized text.
            encoding: Encoding used to decode the file.
            method: Detection method used ("charset-normalizer", "chardet",
                "utf-8-fallback", or "empty-file").

        Returns:
            Dictionary containing document metadata.
        """
        return {
            "source": str(file_path),
            "page_count": 1,
            "format": "text",
            "encoding": encoding,
            "detection_method": method,
            "line_count": len(text.splitlines()),
            "char_count": len(text),
            "word_count": len(text.split()),
        }

    def extract_metadata(self, file_path: Path) -> dict[str, Any]:
        """
        Extract metadata from a text file without returning its content.

        TODO(2.2.5): This will be expanded in step 2.2.5 with the full
        encoding-reporting spec.

        Args:
            file_path: Path to the text file (str or Path).

        Returns:
            Dictionary containing document metadata.

        Raises:
            LoadError: If the file cannot be read or decoded.
        """
        file_path = Path(file_path)
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise LoadError(
                f"Unsupported file extension: {file_path.suffix}. "
                "Expected .txt or .text",
                source_path=str(file_path),
            )

        try:
            data = file_path.read_bytes()
            text, encoding, method = self._decode(data, file_path)
            return self._build_metadata(file_path, text, encoding, method)
        except LoadError:
            raise
        except Exception as e:
            raise LoadError(
                f"Failed to extract metadata from Text: {e}",
                source_path=str(file_path),
                cause=e,
            ) from e

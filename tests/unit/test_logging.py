"""
Unit tests for structured logging (step 2.6.2).

Covers correlation IDs, JSON fields, redaction, level filtering and idempotent
configuration. No global logger is modified outside the tests.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator

import pytest

from src.airag.ingestion import logging as ingestion_logging
from src.airag.ingestion.cli import create_parser
from src.airag.ingestion.logging import (
    clear_correlation_id,
    configure_logging,
    get_correlation_id,
    new_correlation_id,
    set_correlation_id,
)


@pytest.fixture(autouse=True)
def reset_correlation() -> Iterator[None]:
    clear_correlation_id()
    yield
    clear_correlation_id()


def make_logger(
    stream: io.StringIO, level: str = "INFO"
) -> logging.Logger:
    return configure_logging(
        level, stream=stream, logger_name="airag.test.logging", force=True
    )


class TestCorrelationId:
    def test_new_id_is_unique_hex(self) -> None:
        first = new_correlation_id()
        second = new_correlation_id()
        assert first != second
        assert len(first) == 32
        int(first, 16)

    def test_set_get_clear(self) -> None:
        assert set_correlation_id("fixed-id") == "fixed-id"
        assert get_correlation_id() == "fixed-id"
        clear_correlation_id()
        assert get_correlation_id() is None

    def test_set_generates_when_omitted(self) -> None:
        value = set_correlation_id()
        assert value
        assert get_correlation_id() == value


class TestJsonLogging:
    def test_record_fields(self) -> None:
        stream = io.StringIO()
        logger = make_logger(stream)
        set_correlation_id("cid-123")

        logger.info("loaded %s", "doc.txt", extra={"source_path": "doc.txt"})

        payload = json.loads(stream.getvalue().strip())
        assert payload["level"] == "INFO"
        assert payload["message"] == "loaded doc.txt"
        assert payload["logger"] == "airag.test.logging"
        assert payload["correlation_id"] == "cid-123"
        assert payload["source_path"] == "doc.txt"
        assert "timestamp" in payload

    def test_missing_correlation_id_uses_placeholder(self) -> None:
        stream = io.StringIO()
        logger = make_logger(stream)

        logger.info("no correlation")

        assert json.loads(stream.getvalue().strip())["correlation_id"] == "-"

    def test_level_filtering(self) -> None:
        stream = io.StringIO()
        logger = make_logger(stream, level="ERROR")

        logger.info("hidden")
        logger.error("visible")

        assert "hidden" not in stream.getvalue()
        assert json.loads(stream.getvalue().strip())["level"] == "ERROR"

    def test_sensitive_extra_is_redacted(self) -> None:
        stream = io.StringIO()
        logger = make_logger(stream)

        logger.info(
            "credentials",
            extra={
                "password": "hunter2",
                "api_token": "abc123",
                "authorization": "Bearer x",
                "source_path": "doc.txt",
            },
        )

        payload = json.loads(stream.getvalue().strip())
        assert payload["password"] == "***"
        assert payload["api_token"] == "***"
        assert payload["authorization"] == "***"
        assert payload["source_path"] == "doc.txt"

    def test_dict_message_is_redacted(self) -> None:
        stream = io.StringIO()
        logger = make_logger(stream)

        logger.info({"password": "hunter2", "source": "doc.txt"})

        payload = json.loads(stream.getvalue().strip())
        assert payload["password"] == "***"

    def test_configuration_is_idempotent(self) -> None:
        stream = io.StringIO()
        logger = make_logger(stream)
        make_logger(stream)

        handlers = [h for h in logger.handlers if h.name == "airag_json"]
        assert len(handlers) == 1

        logger.info("once")
        assert len(stream.getvalue().strip().splitlines()) == 1

    def test_exception_details_included(self) -> None:
        stream = io.StringIO()
        logger = make_logger(stream)

        try:
            _ = 1 / 0
        except ZeroDivisionError:
            logger.exception("failure")

        assert "ZeroDivisionError" in stream.getvalue()

    def test_reexported_api(self) -> None:
        import src.airag.ingestion as ingestion

        assert ingestion.configure_logging is configure_logging
        assert ingestion.new_correlation_id is new_correlation_id
        assert ingestion.get_correlation_id is get_correlation_id
        assert ingestion.set_correlation_id is set_correlation_id
        assert ingestion.clear_correlation_id is clear_correlation_id
        assert "password" in ingestion_logging.DEFAULT_SENSITIVE_KEYS


class TestCliIntegration:
    def test_parser_accepts_log_level(self) -> None:
        args = create_parser().parse_args(["--log-level", "DEBUG", "extensions"])
        assert args.log_level == "DEBUG"
        assert args.command == "extensions"

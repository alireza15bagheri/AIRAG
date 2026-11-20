"""
Unit tests for the LoaderFactory (step 2.1.4).

The registry is snapshotted around every test so registrations made here never
leak into the built-in registry or into other tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

import src.airag.ingestion.factory as factory
import src.airag.ingestion.loaders as loaders_package
from src.airag.ingestion.exceptions import LoadError, UnsupportedFormatError
from src.airag.ingestion.loaders import (
    DocumentLoader,
    DocxLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    get_loader,
    register_loader,
    supported_extensions,
)

EXPECTED: dict[str, type[DocumentLoader]] = {
    ".docx": DocxLoader,
    ".markdown": MarkdownLoader,
    ".md": MarkdownLoader,
    ".pdf": PDFLoader,
    ".text": TextLoader,
    ".txt": TextLoader,
}


@pytest.fixture(autouse=True)
def preserve_registry() -> Any:
    """Restore the built-in registry after each test."""
    snapshot = dict(factory._registry)
    yield
    factory._registry.clear()
    factory._registry.update(snapshot)


class TestCompleteness:
    """Prove no concrete loader in the package is unreachable."""

    def test_every_concrete_loader_is_registered(self) -> None:
        subclasses: set[type[DocumentLoader]] = set()
        stack: list[type[DocumentLoader]] = list(DocumentLoader.__subclasses__())
        while stack:
            loader_cls = stack.pop()
            if loader_cls in subclasses:
                continue
            subclasses.add(loader_cls)
            stack.extend(loader_cls.__subclasses__())

        concrete = {
            cls
            for cls in subclasses
            if cls.SUPPORTED_EXTENSIONS
            and cls.__module__.startswith("src.airag.ingestion.loaders")
        }
        assert concrete, "expected the four built-in loaders to be discovered"
        registered = set(factory._registry.values())
        assert concrete <= registered, f"unreachable loaders: {concrete - registered}"

        for loader_cls in concrete:
            for extension in loader_cls.SUPPORTED_EXTENSIONS:
                assert type(get_loader(f"sample{extension}")) is loader_cls

    def test_package_reexports_factory_api(self) -> None:
        assert loaders_package.get_loader is get_loader
        assert loaders_package.register_loader is register_loader
        assert loaders_package.supported_extensions is supported_extensions


class TestDefaultRegistration:
    """The built-in registry maps every supported extension correctly."""

    def test_supported_extensions_exact(self) -> None:
        assert supported_extensions() == sorted(EXPECTED)

    @pytest.mark.parametrize(
        ("extension", "loader_cls"), sorted(EXPECTED.items()), ids=sorted(EXPECTED)
    )
    def test_resolves_expected_class(
        self, extension: str, loader_cls: type[DocumentLoader]
    ) -> None:
        assert type(get_loader(f"sample{extension}")) is loader_cls

    @pytest.mark.parametrize("extension", sorted(EXPECTED), ids=sorted(EXPECTED))
    def test_case_insensitive(self, extension: str) -> None:
        assert type(get_loader(f"SAMPLE{extension.upper()}")) is EXPECTED[extension]

    @pytest.mark.parametrize(
        ("extension", "loader_cls"), sorted(EXPECTED.items()), ids=sorted(EXPECTED)
    )
    def test_str_and_path_inputs_equivalent(
        self, extension: str, loader_cls: type[DocumentLoader], tmp_path: Path
    ) -> None:
        path = tmp_path / f"sample{extension}"
        assert type(get_loader(str(path))) is loader_cls
        assert type(get_loader(path)) is loader_cls

    def test_each_call_returns_fresh_instance(self) -> None:
        first = get_loader("first.pdf")
        second = get_loader("second.pdf")
        assert first is not second

    def test_protocol_conformance(self) -> None:
        for extension in sorted(EXPECTED):
            assert isinstance(get_loader(f"sample{extension}"), DocumentLoader)

    def test_debug_log_on_resolution(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG):
            get_loader("logged.PDF")
        assert any("resolved" in record.message.lower() for record in caplog.records)


class TestResolutionErrors:
    """Unknown, missing, and non-file inputs raise UnsupportedFormatError."""

    @pytest.mark.parametrize("path", ["README", ".env", "report.tar.gz", "notes.xyz", "old.doc"])
    def test_unsupported_paths(self, path: str) -> None:
        with pytest.raises(UnsupportedFormatError) as exc_info:
            get_loader(path)

        assert isinstance(exc_info.value, LoadError)
        assert ", ".join(supported_extensions()) in str(exc_info.value)

    def test_directory_path(self, tmp_path: Path) -> None:
        with pytest.raises(UnsupportedFormatError) as exc_info:
            get_loader(tmp_path)

        message = str(exc_info.value).lower()
        assert "directory" in message
        assert "2.5.1" in message

    def test_nonexistent_path_with_known_extension_still_resolves(self, tmp_path: Path) -> None:
        result = get_loader(tmp_path / "missing.pdf")
        assert isinstance(result, PDFLoader)


class TestRegistrationApi:
    """register_loader is centralized, validated, fail-fast, and overridable."""

    def _stub_loader(self, *extensions: str) -> type[DocumentLoader]:
        @register_loader(*extensions)
        class StubLoader(DocumentLoader):
            SUPPORTED_EXTENSIONS = frozenset(extensions)

            def load(self, file_path: Path) -> None:
                raise NotImplementedError

        return StubLoader

    def test_decorator_registers_and_resolves(self) -> None:
        stub = self._stub_loader(".stub")
        assert type(get_loader("data.stub")) is stub
        assert ".stub" in supported_extensions()

    def test_extensions_are_normalized(self) -> None:
        stub = self._stub_loader("STUB2")
        assert type(get_loader("data.stub2")) is stub
        assert ".stub2" in supported_extensions()

    def test_duplicate_registration_raises(self) -> None:
        stub = self._stub_loader(".stub")
        with pytest.raises(ValueError, match="already registered"):
            register_loader(".pdf")(stub)

    def test_override_registration_warns_and_replaces(self, caplog: pytest.LogCaptureFixture) -> None:
        stub = self._stub_loader(".stub")
        with caplog.at_level(logging.WARNING):
            register_loader(".pdf", override=True)(stub)

        assert factory._registry[".pdf"] is stub
        assert any("overriding" in record.message.lower() for record in caplog.records)

    def test_same_class_reregistration_is_allowed(self) -> None:
        stub = self._stub_loader(".stub")
        register_loader(".stub")(stub)
        assert factory._registry[".stub"] is stub

    def test_non_loader_class_rejected(self) -> None:
        with pytest.raises(TypeError, match="DocumentLoader"):
            register_loader(".broken")(object)

    def test_duck_typed_loader_accepted(self) -> None:
        @register_loader(".duck")
        class DuckLoader:
            def load(self, file_path: Path) -> None:
                raise NotImplementedError

        assert isinstance(get_loader("data.duck"), DuckLoader)

    def test_empty_extension_list_rejected(self) -> None:
        stub = self._stub_loader(".stub")
        with pytest.raises(ValueError, match="(?i)at least one extension"):
            register_loader()(stub)

    def test_loader_options_are_forwarded(self) -> None:
        captured: dict[str, Any] = {}

        @register_loader(".opts")
        class OptionsLoader(DocumentLoader):
            SUPPORTED_EXTENSIONS = frozenset({".opts"})

            def __init__(
                self, config: dict[str, Any] | None = None, **options: Any
            ) -> None:
                super().__init__(config)
                captured.update(options)

            def load(self, file_path: Path) -> None:
                raise NotImplementedError

        get_loader("data.opts", password="secret", encoding="utf-8")
        assert captured == {"password": "secret", "encoding": "utf-8"}

    def test_existing_registration_survives_teardown(self) -> None:
        assert type(get_loader("kept.pdf")) is PDFLoader

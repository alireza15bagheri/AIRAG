"""
CLI interface for document ingestion.

Step 2.5.x: Command-line interface for the ingestion pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.airag.ingestion.config import create_components, load_config
from src.airag.ingestion.dlq import DeadLetterQueue
from src.airag.ingestion.logging import configure_logging, set_correlation_id
from src.airag.ingestion.metrics import (
    CHUNKS_CREATED,
    FILES_FAILED,
    FILES_PROCESSED,
    MetricsRegistry,
)
from src.airag.ingestion.pipeline import (
    CheckpointManager,
    IngestionPipeline,
    TqdmProgress,
)
from src.airag.ingestion.service import (
    DocumentIngestionService,
    create_ingestion_service,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the ingestion CLI."""
    parser = argparse.ArgumentParser(
        prog="airag-ingest",
        description="AIRAG Document Ingestion CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Minimum level for structured JSON logs on stderr",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest directory command
    dir_parser = subparsers.add_parser(
        "directory",
        help="Ingest all supported documents from a directory",
    )
    dir_parser.add_argument("path", type=Path, help="Directory path to ingest")
    dir_parser.add_argument(
        "-r", "--recursive", action="store_true", default=True,
        help="Recursively traverse subdirectories"
    )
    dir_parser.add_argument(
        "--no-recursive", dest="recursive", action="store_false",
        help="Only process top-level directory"
    )
    dir_parser.add_argument(
        "-e", "--extensions", nargs="+",
        help="File extensions to process (default: all supported)"
    )
    dir_parser.add_argument(
        "--no-deduplicate", action="store_true",
        help="Disable content-based deduplication"
    )
    dir_parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop on first error instead of skipping"
    )
    dir_parser.add_argument(
        "-o", "--output", type=Path,
        help="Output results to JSON file"
    )
    dir_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output"
    )

    # Ingest files command
    files_parser = subparsers.add_parser(
        "files",
        help="Ingest specific files",
    )
    files_parser.add_argument("paths", nargs="+", type=Path, help="File paths to ingest")
    files_parser.add_argument(
        "--no-deduplicate", action="store_true",
        help="Disable content-based deduplication"
    )
    files_parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop on first error instead of skipping"
    )
    files_parser.add_argument(
        "-o", "--output", type=Path,
        help="Output results to JSON file"
    )
    files_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output"
    )

    # Full pipeline ingestion (step 2.5.2)
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Full pipeline: load, chunk, embed and store a file or directory",
    )
    ingest_parser.add_argument(
        "path", type=Path, nargs="?",
        help="File or directory to ingest (not needed with --retry-dlq)"
    )
    ingest_parser.add_argument(
        "--config", type=Path, help="YAML/JSON config file (config/ingestion.yaml)"
    )
    ingest_parser.add_argument(
        "--chunker", choices=("fixed", "sentence", "semantic"),
        help="Override the chunking strategy from config"
    )
    ingest_parser.add_argument(
        "--chunk-size", type=int, help="Override chunk_size from config"
    )
    ingest_parser.add_argument(
        "--overlap", type=int, help="Override overlap from config"
    )
    ingest_parser.add_argument(
        "--no-recursive", action="store_true",
        help="Only process top-level files of a directory"
    )
    ingest_parser.add_argument(
        "--checkpoint", type=Path,
        help="JSON checkpoint file enabling resume of completed documents"
    )
    ingest_parser.add_argument(
        "--progress", action="store_true",
        help="Show tqdm progress on stderr (stdout stays JSON-clean)"
    )
    ingest_parser.add_argument(
        "--dlq", type=Path, help="JSONL dead letter queue path (overrides config)"
    )
    ingest_parser.add_argument(
        "--retry-dlq", action="store_true",
        help="Re-ingest pending DLQ sources instead of a new path"
    )
    ingest_parser.add_argument(
        "-o", "--output", type=Path, help="Write pipeline results to JSON file"
    )
    ingest_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose JSON output"
    )

    # List supported extensions
    subparsers.add_parser(
        "extensions",
        help="List supported file extensions",
    )

    return parser


def format_result(result) -> str:
    """Format ingestion result for human-readable output."""
    lines = [
        "Ingestion Complete",
        f"  Loaded:    {result.loaded}",
        f"  Failed:    {result.failed}",
        f"  Skipped:   {result.skipped}",
        f"  Started:   {result.started_at}",
        f"  Completed: {result.completed_at}",
    ]

    if result.failed > 0:
        lines.append("\nFailed files:")
        for r in result.results:
            if not r.success:
                lines.append(f"  - {r.error}")

    return "\n".join(lines)


def run_directory(args: argparse.Namespace) -> int:
    """Run directory ingestion."""
    service = create_ingestion_service(
        extensions=tuple(args.extensions) if args.extensions else None,
        recursive=args.recursive,
        skip_errors=not args.fail_fast,
        deduplicate=not args.no_deduplicate,
    )

    try:
        result = service.ingest_directory(args.path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(json.dumps({
            "loaded": result.loaded,
            "failed": result.failed,
            "skipped": result.skipped,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "results": [
                {
                    "success": r.success,
                    "error": r.error,
                    "source": r.document.metadata.get("source") if r.document else None,
                    "hash": r.document.content_hash[:16] + "..." if r.document else None,
                }
                for r in result.results
            ],
        }, indent=2))
    else:
        print(format_result(result))

    if args.output:
        args.output.write_text(json.dumps({
            "loaded": result.loaded,
            "failed": result.failed,
            "skipped": result.skipped,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "results": [
                {
                    "success": r.success,
                    "error": r.error,
                    "source": r.document.metadata.get("source") if r.document else None,
                    "hash": r.document.content_hash if r.document else None,
                }
                for r in result.results
            ],
        }, indent=2))
        print(f"\nResults written to {args.output}")

    return 0 if result.failed == 0 else 1


def run_files(args: argparse.Namespace) -> int:
    """Run file ingestion."""
    service = create_ingestion_service(
        deduplicate=not args.no_deduplicate,
        skip_errors=not args.fail_fast,
    )

    try:
        result = service.ingest_files(args.paths)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(json.dumps({
            "loaded": result.loaded,
            "failed": result.failed,
            "skipped": result.skipped,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "results": [
                {
                    "success": r.success,
                    "error": r.error,
                    "source": r.document.metadata.get("source") if r.document else None,
                    "hash": r.document.content_hash[:16] + "..." if r.document else None,
                }
                for r in result.results
            ],
        }, indent=2))
    else:
        print(format_result(result))

    if args.output:
        args.output.write_text(json.dumps({
            "loaded": result.loaded,
            "failed": result.failed,
            "skipped": result.skipped,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "results": [
                {
                    "success": r.success,
                    "error": r.error,
                    "source": r.document.metadata.get("source") if r.document else None,
                    "hash": r.document.content_hash if r.document else None,
                }
                for r in result.results
            ],
        }, indent=2))
        print(f"\nResults written to {args.output}")

    return 0 if result.failed == 0 else 1


def format_pipeline_result(result, metrics: MetricsRegistry | None = None) -> str:
    """Format a full-pipeline result for human-readable output."""
    lines = [
        "Pipeline Complete",
        f"  Files:     {result.files}",
        f"  Succeeded: {result.succeeded}",
        f"  Failed:    {result.failed}",
        f"  Chunks:    {result.chunks}",
        f"  Written:   {result.written}",
        f"  Updated:   {result.updated}",
        f"  Skipped:   {result.skipped}",
        f"  Resumed:   {result.resumed}",
    ]
    if metrics is not None:
        lines.append(
            "  Metrics:   "
            f"{metrics.get_counter(FILES_PROCESSED)} files processed, "
            f"{metrics.get_counter(CHUNKS_CREATED)} chunks created, "
            f"{metrics.get_counter(FILES_FAILED)} failed"
        )
    if result.failed:
        lines.append("\nFailed files:")
        for item in result.file_results:
            if not item.success:
                lines.append(f"  - {item.source}: {item.error}")
    return "\n".join(lines)


def run_ingest(args: argparse.Namespace) -> int:
    """Run the full ingestion pipeline with optional YAML/JSON config."""
    try:
        config = load_config(args.config)
        if args.chunker is not None:
            config.chunker = args.chunker
        if args.chunk_size is not None:
            config.chunk_size = args.chunk_size
        if args.overlap is not None:
            config.overlap = args.overlap
        if args.no_recursive:
            config.recursive = False
        config.validate()

        dlq_path = args.dlq or (
            Path(config.dlq_path) if config.dlq_path is not None else None
        )
        if args.retry_dlq and dlq_path is None:
            raise ValueError("--retry-dlq requires --dlq or config.dlq_path")
        if not args.retry_dlq and args.path is None:
            raise ValueError("a path is required unless --retry-dlq is used")
        dlq = DeadLetterQueue(dlq_path) if dlq_path is not None else None

        chunker, embedding_pipeline = create_components(config)
        metrics = getattr(embedding_pipeline, "metrics", None) or MetricsRegistry()
        service = DocumentIngestionService(
            recursive=config.recursive,
            loader_options=config.loader_options,
        )
        checkpoint = CheckpointManager(args.checkpoint) if args.checkpoint else None
        progress = TqdmProgress() if args.progress else None
        pipeline = IngestionPipeline(
            chunker,
            embedding_pipeline,
            service,
            checkpoint=checkpoint,
            progress=progress,
            metrics=metrics,
            dlq=dlq,
        )
        result = (
            pipeline.retry_dlq()
            if args.retry_dlq
            else pipeline.ingest(args.path)
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    payload = {
        "files": result.files,
        "succeeded": result.succeeded,
        "failed": result.failed,
        "chunks": result.chunks,
        "written": result.written,
        "updated": result.updated,
        "skipped": result.skipped,
        "resumed": result.resumed,
        "metrics": metrics.snapshot(),
        "dlq": (
            {"path": str(dlq.path), "pending": len(dlq.pending_sources())}
            if dlq is not None
            else None
        ),
        "results": [
            {
                "source": item.source,
                "success": item.success,
                "chunks": item.chunks,
                "written": item.written,
                "updated": item.updated,
                "skipped": item.skipped,
                "resumed": item.resumed,
                "error": item.error,
            }
            for item in result.file_results
        ],
    }
    if args.verbose:
        print(json.dumps(payload, indent=2))
    else:
        print(format_pipeline_result(result, metrics))

    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nResults written to {args.output}")
    return (
        0
        if result.failed == 0 and (result.succeeded + result.resumed) > 0
        else 1
    )


def run_extensions(_: argparse.Namespace) -> int:
    """List supported extensions."""
    from src.airag.ingestion.factory import supported_extensions
    exts = supported_extensions()
    print("Supported extensions:")
    for ext in sorted(exts):
        print(f"  {ext}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    set_correlation_id()

    if args.command == "directory":
        return run_directory(args)
    elif args.command == "files":
        return run_files(args)
    elif args.command == "ingest":
        return run_ingest(args)
    elif args.command == "extensions":
        return run_extensions(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

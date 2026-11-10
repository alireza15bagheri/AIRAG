#!/usr/bin/env python
"""
Session 1.1 Demo: LLMs, Transformers, and Tokens
=================================================
Educational demonstration of tokenization concepts.
Run: python demo/session_1_1_tokenization.py
"""

from __future__ import annotations

import sys


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def demo_tiktoken_basic() -> None:
    """Demonstrate basic tiktoken usage (OpenAI GPT tokenizer)."""
    print_section("1. TIKTOKEN - OpenAI GPT Tokenizer")

    try:
        import tiktoken
    except ImportError:
        print("tiktoken not installed. Run: pip install tiktoken")
        return

    # Load the GPT-4 / GPT-3.5-turbo tokenizer (cl100k_base)
    enc = tiktoken.get_encoding("cl100k_base")
    print(f"Encoding name: {enc.name}")
    print(f"Vocabulary size: {enc.n_vocab:,} tokens")
    print(f"Special tokens: {enc.special_tokens}")

    # Example texts
    texts = [
        "Hello, RAG world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Hello world",  # Short
        "A" * 100,      # Repeated character
        "این یک عبارت فارسی است",
        "سلام دنیا به زبان پارسی!",
        "شاپ واسه امروز بسه!",
    ]

    for text in texts:
        tokens = enc.encode(text)
        decoded = enc.decode(tokens)
        print(f"\n  Text:     '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"  Tokens:   {tokens}")
        print(f"  Count:    {len(tokens)} tokens")
        print(f"  Decoded:  '{decoded}'")
        print(f"  Match:    {text == decoded}")


def demo_token_visualization() -> None:
    """Visualize individual token boundaries."""
    print_section("2. TOKEN VISUALIZATION - See Boundaries")

    try:
        import tiktoken
    except ImportError:
        return

    enc = tiktoken.get_encoding("cl100k_base")

    text = "Hello, world! How are you?"
    tokens = enc.encode(text)

    print(f"Text: '{text}'")
    print(f"Total tokens: {len(tokens)}")
    print("\nToken-by-token breakdown:")
    print("-" * 40)

    for i, token_id in enumerate(tokens):
        token_bytes = enc.decode_single_token_bytes(token_id)
        token_str = token_bytes.decode("utf-8", errors="replace")
        # Show visible representation
        visible = repr(token_str)
        print(f"  [{i:2d}] ID: {token_id:6d}  Bytes: {token_bytes!r:12s}  ->  {visible}")


def demo_context_window_math() -> None:
    """Calculate context window usage."""
    print_section("3. CONTEXT WINDOW MATH")

    try:
        import tiktoken
    except ImportError:
        return

    enc = tiktoken.get_encoding("cl100k_base")

    # Sample document
    doc = """
    Retrieval-Augmented Generation (RAG) is a technique that enhances
    large language models by retrieving relevant documents from a knowledge
    base and providing them as context for generation. This approach
    combines the strengths of retrieval systems and generative models.
    """ * 10  # Repeat to make it longer

    tokens = enc.encode(doc)
    token_count = len(tokens)

    # Common context windows
    windows = {
        "GPT-3.5": 4_096,
        "GPT-4": 8_192,
        "GPT-4 Turbo": 128_000,
        "LLaMA-3.1": 128_000,
        "Gemini 1.5": 1_000_000,
    }

    print(f"Document token count: {token_count:,}")
    print(f"Document char count:  {len(doc):,}")
    print(f"Ratio (chars/token):  {len(doc) / token_count:.2f}")
    print()

    for model, window in windows.items():
        pct = (token_count / window) * 100
        status = "✅ Fits" if token_count < window else "❌ Exceeds"
        print(f"  {model:15s} ({window:>8,} tokens): {pct:5.1f}%  {status}")

    # Show how many such docs fit
    for model, window in windows.items():
        if token_count > 0:
            count = window // token_count
            print(f"  {model:15s}: ~{count} similar documents fit in context")


def demo_huggingface_tokenizer() -> None:
    """Compare with HuggingFace tokenizer (LLaMA style)."""
    print_section("4. HUGGINGFACE TOKENIZER - LLaMA-3 Style")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers not installed. Run: pip install transformers")
        return

    print("Loading LLaMA-3.2-3B tokenizer (this downloads on first run)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    except Exception as e:
        print(f"Could not load LLaMA tokenizer (may need auth): {e}")
        print("Using GPT-2 tokenizer as fallback...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {tokenizer.vocab_size:,}")

    texts = [
        "Hello, RAG world!",
        "The quick brown fox jumps.",
        "Machine learning 🤖",
        "این یک عبارت فارسی است",
    ]

    for text in texts:
        # encode returns list of token IDs
        tokens = tokenizer.encode(text)
        # decode converts back to string
        decoded = tokenizer.decode(tokens)

        print(f"\n  Text:     '{text}'")
        print(f"  Tokens:   {tokens}")
        print(f"  Count:    {len(tokens)}")
        print(f"  Decoded:  '{decoded}'")

    # Show special tokens
    print("\nSpecial tokens:")
    for name, id_ in tokenizer.special_tokens_map.items():
        print(f"  {name}: {id_}")


def demo_tokenizer_comparison() -> None:
    """Compare tokenizers on same text."""
    print_section("5. TOKENIZER COMPARISON")

    try:
        import tiktoken
        from transformers import AutoTokenizer
    except ImportError:
        return

    enc_openai = tiktoken.get_encoding("cl100k_base")
    try:
        tokenizer_hf = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    except Exception:
        tokenizer_hf = AutoTokenizer.from_pretrained("gpt2")

    test_texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning & AI",
        "Привет мир",  # Cyrillic
        "你好世界",     # Chinese
        "🚀🤖🌟",       # Emojis
        "def hello():\n    return 'world'",  # Code
        "این یک عبارت فارسی است",  # Persian
    ]

    print(f"{'Text':<40s} {'OpenAI':>8s} {'HF/LLaMA':>10s} {'Diff':>6s}")
    print("-" * 70)

    for text in test_texts:
        openai_tokens = len(enc_openai.encode(text))
        hf_tokens = len(tokenizer_hf.encode(text))
        diff = hf_tokens - openai_tokens
        diff_str = f"{diff:+d}" if diff != 0 else "  --"
        display = text[:37] + "..." if len(text) > 40 else text
        print(f"'{display:<40s}' {openai_tokens:>8d} {hf_tokens:>10d} {diff_str:>6s}")


def main() -> None:
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  SESSION 1.1: LLMs, TRANSFORMERS, AND TOKENS")
    print("  Educational Demo - Tokenization Deep Dive")
    print("=" * 60)

    demos = [
        demo_tiktoken_basic,
        demo_token_visualization,
        demo_context_window_math,
        demo_huggingface_tokenizer,
        demo_tokenizer_comparison,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n⚠️  Demo skipped: {e}")

    print("\n" + "=" * 60)
    print("  SESSION 1.1 COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  • Tokens ≠ words (1 token ≈ 0.75 English words)")
    print("  • Different tokenizers produce different token counts")
    print("  • Context window limits how much context you can provide")
    print("  • Tokenization affects chunking strategy in RAG")
    print("  • Special tokens (BOS, EOS, PAD) control generation")


if __name__ == "__main__":
    main()
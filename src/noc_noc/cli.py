import argparse
import hashlib
import os
from functools import cache
from pathlib import Path

import httpx

from noc_noc.application.benchmark import Case, RetrieverFactory, evaluate, split
from noc_noc.application.rerank import Reranker
from noc_noc.domain.classification import Classification
from noc_noc.infrastructure import statcan
from noc_noc.infrastructure.claude import ClaudeRankModel
from noc_noc.infrastructure.dense import DenseRetriever, FastEmbedder, HybridRetriever
from noc_noc.infrastructure.lexical import TfidfRetriever

CACHE_DIR = Path(os.environ.get("NOC_CACHE", "data/cache"))


@cache
def embedder() -> FastEmbedder:
    return FastEmbedder(CACHE_DIR)


@cache
def claude() -> ClaudeRankModel:
    import anthropic  # noqa: PLC0415 (only the rerank method needs it)

    return ClaudeRankModel(anthropic.Anthropic())


def hybrid(c: Classification) -> HybridRetriever:
    return HybridRetriever([TfidfRetriever(c), DenseRetriever(c, embedder())])


METHODS: dict[str, RetrieverFactory] = {
    "tfidf": TfidfRetriever,
    "dense": lambda c: DenseRetriever(c, embedder()),
    "hybrid": hybrid,
    "rerank": lambda c: Reranker(hybrid(c), c, claude()),
}
PAID_METHODS = {"rerank"}
DEFAULT_PAID_SAMPLE = 200  # about $3.50 with claude-opus-5-5 at low effort


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="noc-noc")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="find the NOC codes for a job title")
    search.add_argument("query")
    search.add_argument("--method", choices=METHODS, default="hybrid")
    search.add_argument("-k", type=int, default=5)
    benchmark = commands.add_parser("eval", help="score methods on the held-out benchmark")
    benchmark.add_argument("--method", choices=METHODS, action="append", dest="methods")
    benchmark.add_argument(
        "--sample",
        type=int,
        help=f"score a fixed sample of titles (default for rerank: {DEFAULT_PAID_SAMPLE})",
    )
    args = parser.parse_args(argv)

    with httpx.Client(timeout=60) as client:
        classification = statcan.load(CACHE_DIR, client)
    paid = PAID_METHODS & ({args.method} if args.command == "search" else set(args.methods or []))
    try:
        run(args, classification)
    except TypeError as error:
        if not paid:
            raise
        # The SDK raises TypeError when it finds no credentials at request time
        raise SystemExit(
            f"{error}\n\nThe rerank method calls Claude. Set ANTHROPIC_API_KEY or run "
            "`ant auth login`, then try again."
        ) from error


def run(args: argparse.Namespace, classification: Classification) -> None:
    if args.command == "search":
        print(run_search(classification, args.method, args.query, args.k))
    else:
        methods = args.methods or [m for m in METHODS if m not in PAID_METHODS]
        sample = args.sample
        if sample is None and PAID_METHODS & set(methods):
            sample = DEFAULT_PAID_SAMPLE
        print(run_eval(classification, methods, sample))


def run_search(classification: Classification, method: str, query: str, k: int) -> str:
    retriever = METHODS[method](classification)
    if isinstance(retriever, Reranker):
        lines = []
        for rank, match in enumerate(retriever.explain(query, k), start=1):
            o = match.occupation
            lines.append(f"{rank}. {o.code}  {o.title}  (TEER {o.teer.value})")
            lines += [f"     ✓ {duty}" for duty in match.matching_duties]
        return "\n".join(lines) or "No match."
    lines = []
    for rank, code in enumerate(retriever.search(query, k), start=1):
        occupation = classification.get(code)
        assert occupation is not None
        lines.append(f"{rank}. {code}  {occupation.title}  (TEER {occupation.teer.value})")
    return "\n".join(lines) or "No match."


def sample_cases(cases: list[Case], size: int | None) -> list[Case]:
    """A fixed sample: the cases whose text hashes lowest, so every run scores the same titles."""
    if size is None:
        return cases
    return sorted(cases, key=lambda c: hashlib.sha256(c.query.encode()).hexdigest())[:size]


def run_eval(classification: Classification, methods: list[str], sample: int | None = None) -> str:
    index, all_cases = split(classification)
    cases = sample_cases(all_cases, sample)
    rows = [
        f"{len(cases):,} held-out titles"
        + (f" (fixed sample of {len(all_cases):,})" if sample else "")
        + "\n",
        "| Method | top-1 | top-5 | MRR | Confusable errors |",
        "|---|---|---|---|---|",
    ]
    for name in methods:
        r = evaluate(METHODS[name](index), cases, index)
        confusable = f"{r.confusable_error_share:.0%}"
        rows.append(f"| {name} | {r.top1:.1%} | {r.top5:.1%} | {r.mrr:.3f} | {confusable} |")
    if PAID_METHODS & set(methods):
        usage = claude().usage
        rows.append(
            f"\nClaude: {usage.requests} requests, {usage.input_tokens:,} input and "
            f"{usage.output_tokens:,} output tokens, ${usage.cost_usd:.2f}"
        )
    return "\n".join(rows)

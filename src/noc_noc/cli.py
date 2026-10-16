import argparse
import os
from functools import cache
from pathlib import Path

import httpx

from noc_noc.application.benchmark import RetrieverFactory, evaluate, split
from noc_noc.domain.classification import Classification
from noc_noc.infrastructure import statcan
from noc_noc.infrastructure.dense import DenseRetriever, FastEmbedder, HybridRetriever
from noc_noc.infrastructure.lexical import TfidfRetriever

CACHE_DIR = Path(os.environ.get("NOC_CACHE", "data/cache"))


@cache
def embedder() -> FastEmbedder:
    return FastEmbedder(CACHE_DIR)


METHODS: dict[str, RetrieverFactory] = {
    "tfidf": TfidfRetriever,
    "dense": lambda c: DenseRetriever(c, embedder()),
    "hybrid": lambda c: HybridRetriever([TfidfRetriever(c), DenseRetriever(c, embedder())]),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="noc-noc")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="find the NOC codes for a job title")
    search.add_argument("query")
    search.add_argument("--method", choices=METHODS, default="hybrid")
    search.add_argument("-k", type=int, default=5)
    benchmark = commands.add_parser("eval", help="score methods on the held-out benchmark")
    benchmark.add_argument("--method", choices=METHODS, action="append", dest="methods")
    args = parser.parse_args(argv)

    with httpx.Client(timeout=60) as client:
        classification = statcan.load(CACHE_DIR, client)
    if args.command == "search":
        print(run_search(classification, args.method, args.query, args.k))
    else:
        print(run_eval(classification, args.methods or list(METHODS)))


def run_search(classification: Classification, method: str, query: str, k: int) -> str:
    retriever = METHODS[method](classification)
    lines = []
    for rank, code in enumerate(retriever.search(query, k), start=1):
        occupation = classification.get(code)
        assert occupation is not None
        lines.append(f"{rank}. {code}  {occupation.title}  (TEER {occupation.teer.value})")
    return "\n".join(lines) or "No match."


def run_eval(classification: Classification, methods: list[str]) -> str:
    index, cases = split(classification)
    rows = [
        f"{len(cases):,} held-out titles\n",
        "| Method | top-1 | top-5 | MRR | Confusable errors |",
        "|---|---|---|---|---|",
    ]
    for name in methods:
        r = evaluate(METHODS[name](index), cases, index)
        confusable = f"{r.confusable_error_share:.0%}"
        rows.append(f"| {name} | {r.top1:.1%} | {r.top5:.1%} | {r.mrr:.3f} | {confusable} |")
    return "\n".join(rows)

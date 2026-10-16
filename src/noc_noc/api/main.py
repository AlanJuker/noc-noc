"""Composition root: `uvicorn noc_noc.api.main:build_app --factory`."""

import os

import httpx
from fastapi import FastAPI

from noc_noc.api.app import create_app
from noc_noc.application.rerank import Reranker
from noc_noc.infrastructure import statcan
from noc_noc.infrastructure.dense import DenseRetriever, FastEmbedder, HybridRetriever
from noc_noc.infrastructure.lexical import TfidfRetriever
from noc_noc.infrastructure.settings import CACHE_DIR


def build_app() -> FastAPI:
    with httpx.Client(timeout=60) as client:
        classification = statcan.load(CACHE_DIR, client)
    hybrid = HybridRetriever(
        [TfidfRetriever(classification), DenseRetriever(classification, FastEmbedder(CACHE_DIR))]
    )
    reranker = None
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        import anthropic  # noqa: PLC0415 (only needed when explanations are enabled)

        from noc_noc.infrastructure.claude import ClaudeRankModel  # noqa: PLC0415

        reranker = Reranker(hybrid, classification, ClaudeRankModel(anthropic.Anthropic()))
    return create_app(classification, hybrid, reranker)

from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from noc_noc.application.benchmark import Retriever
from noc_noc.application.rerank import Match, Reranker
from noc_noc.domain.classification import Classification
from noc_noc.domain.occupation import Occupation

TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")
MAX_QUERY = 2_000


class OccupationOut(BaseModel):
    code: str
    title: str
    teer: int
    teer_description: str
    definition: str
    example_titles: list[str]
    main_duties: list[str]
    requirements: list[str]

    @classmethod
    def of(cls, o: Occupation) -> "OccupationOut":
        return cls(
            code=o.code,
            title=o.title,
            teer=o.teer.value,
            teer_description=o.teer.description,
            definition=o.definition,
            example_titles=list(o.example_titles),
            main_duties=list(o.main_duties),
            requirements=list(o.requirements),
        )


class MatchOut(BaseModel):
    code: str
    title: str
    teer: int
    teer_description: str
    matching_duties: list[str]


def create_app(
    classification: Classification, hybrid: Retriever, reranker: Reranker | None = None
) -> FastAPI:
    """`reranker` is optional: without an Anthropic key the app serves the hybrid ranking."""
    app = FastAPI(title="noc-noc", summary="Find your NOC 2021 occupation code.", version="0.1.0")

    def matches(query: str, k: int, explain: bool) -> list[Match]:
        if explain and reranker is not None:
            return reranker.explain(query, k)
        return [Match(o, ()) for c in hybrid.search(query, k) if (o := classification.get(c))]

    @app.get("/api/search")
    def search(
        q: Annotated[str, Query(min_length=2, max_length=MAX_QUERY)],
        k: Annotated[int, Query(ge=1, le=10)] = 5,
        explain: bool = False,
    ) -> list[MatchOut]:
        return [
            MatchOut(
                code=m.occupation.code,
                title=m.occupation.title,
                teer=m.occupation.teer.value,
                teer_description=m.occupation.teer.description,
                matching_duties=list(m.matching_duties),
            )
            for m in matches(q, k, explain)
        ]

    @app.get("/api/occupations/{code}", responses={404: {"description": "Unknown code"}})
    def occupation(code: str) -> OccupationOut:
        found = classification.get(code)
        if found is None:
            raise HTTPException(404, f"No NOC 2021 unit group {code}")
        return OccupationOut.of(found)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request, "index.html", {"explain_available": reranker is not None}
        )

    @app.post("/search", response_class=HTMLResponse, include_in_schema=False)
    def search_page(
        request: Request,
        q: Annotated[str, Form(min_length=2, max_length=MAX_QUERY)],
        explain: Annotated[Literal["on"] | None, Form()] = None,
    ) -> HTMLResponse:
        context = {
            "query": q,
            "matches": matches(q, 5, explain == "on"),
            "explain_available": reranker is not None,
        }
        # HTMX swaps in just the results; a plain form post gets the whole page
        template = "results.html" if request.headers.get("HX-Request") else "index.html"
        return TEMPLATES.TemplateResponse(request, template, context)

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app

# Contributing

1. Open an issue describing the change.
2. Branch from `main` (`feat/…`, `fix/…`, `docs/…`).
3. Commit with [Conventional Commits](https://www.conventionalcommits.org/).
4. Run `uv run ruff check . && uv run mypy && uv run pytest`.
5. If the change affects search quality, run `uv run noc-noc eval` and put the before/after numbers in the PR.
6. Open a pull request that references the issue.

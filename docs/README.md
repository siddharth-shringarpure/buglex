# Documentation Workspace

This directory holds the project's supporting reports as LaTeX source files and figures.

It is split into:

- `report/`: the main research report draft
- `requirements/`: material that will become `requirements.pdf`
- `manual/`: material that will become `manual.pdf`
- `replication/`: material that will become `replication.pdf`

To build all deliverables, use:

```bash
uv run python -m src.tools.build_docs
```

To build only the main report (including tables and figures), use:

```bash
uv run python -m src.tools.build_docs --report-only
```

To build only the support reports, use:

```bash
uv run python -m src.tools.build_docs --support-only
```

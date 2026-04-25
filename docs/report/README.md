# Main Report Draft

This folder holds the main report source files and LaTeX tables used.

The report covers:

- Introduction
- Related Work
- Solution
- Setup
- Experiments
- Limitations
- Future Work
- Conclusion
- Artefact
- References

Build it from the repository root with:

```bash
uv run python -m src.tools.build_docs --report-only
```

This script regenerates the LaTeX table fragments from `results/`, generates the plots, and then rebuilds `docs/report/report.pdf`.

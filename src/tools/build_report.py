"""Generate report tables and rebuild the LaTeX report PDF."""

from pathlib import Path
import shutil
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT_DIR / "docs" / "report"
TABLE_SCRIPT = ROOT_DIR / "src" / "tools" / "make_report_tables.py"
REPORT_TEX = REPORT_DIR / "report.tex"


def main() -> None:
    """Re-generate report tables and compile the LaTeX report."""
    _run([sys.executable, str(TABLE_SCRIPT)], cwd=ROOT_DIR)  # (re-)build tables

    if shutil.which("latexmk"):
        _run(
            [
                "latexmk",
                "-g",
                "-pdf",
                "-interaction=nonstopmode",
                REPORT_TEX.name,
            ],
            cwd=REPORT_DIR,
        )
        return

    if not shutil.which("pdflatex") or not shutil.which("bibtex"):
        raise SystemExit(
            "Could not find latexmk or the pdflatex/bibtex fallback in PATH."
        )

    # First pass to write data needed by BibTeX.
    _run(
        ["pdflatex", "-interaction=nonstopmode", REPORT_TEX.name],
        cwd=REPORT_DIR,
    )
    _run(["bibtex", "report"], cwd=REPORT_DIR)  # build bibliography

    # Two more passes to settle bib entries and cross refs
    _run(
        ["pdflatex", "-interaction=nonstopmode", REPORT_TEX.name],
        cwd=REPORT_DIR,
    )
    _run(
        ["pdflatex", "-interaction=nonstopmode", REPORT_TEX.name],
        cwd=REPORT_DIR,
    )


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    main()

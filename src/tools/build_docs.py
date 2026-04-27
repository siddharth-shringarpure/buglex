"""Build the report PDF and the support PDFs required by the brief."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn

from ..config import DOCS_DIR, REPO_ROOT, REPORT_DIR


_console = Console()


REPORT_TEX = REPORT_DIR / "report.tex"
SUPPORT_DOCS = (
    ("requirements", "requirements.tex", "requirements.pdf"),
    ("manual", "manual.tex", "manual.pdf"),
    ("replication", "replication.tex", "replication.pdf"),
)


def _run(command: list[str], cwd: Path, quiet: bool = False) -> None:
    """Run a command, printing its output unless quiet."""
    if not quiet:
        subprocess.run(command, cwd=cwd, check=True)
        return
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, command, result.stdout, result.stderr
        )


def _build_report() -> None:
    """Re-generate report tables and compile the LaTeX report."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=_console,
        transient=True,
    ) as progress:
        steps = [
            (
                "Generating plots",
                [sys.executable, "-m", "src.plot_results"],
                REPO_ROOT,
                False,
            ),
            (
                "Building tables",
                [sys.executable, "-m", "src.tools.make_report_tables"],
                REPO_ROOT,
                False,
            ),
        ]

        if shutil.which("latexmk"):
            steps.append(
                (
                    "Compiling report (latexmk)",
                    [
                        "latexmk",
                        "-g",
                        "-pdf",
                        "-interaction=nonstopmode",
                        REPORT_TEX.name,
                    ],
                    REPORT_DIR,
                    True,
                )
            )
        elif shutil.which("pdflatex") and shutil.which("bibtex"):
            steps += [
                (
                    "pdflatex pass 1",
                    ["pdflatex", "-interaction=nonstopmode", REPORT_TEX.name],
                    REPORT_DIR,
                    True,
                ),
                ("bibtex", ["bibtex", "report"], REPORT_DIR, True),
                (
                    "pdflatex pass 2 (settle bib entries and cross refs)",
                    ["pdflatex", "-interaction=nonstopmode", REPORT_TEX.name],
                    REPORT_DIR,
                    True,
                ),
                (
                    "pdflatex pass 3 (settle bib entries and cross refs)",
                    ["pdflatex", "-interaction=nonstopmode", REPORT_TEX.name],
                    REPORT_DIR,
                    True,
                ),
            ]
        else:
            raise SystemExit(
                "Could not find latexmk or the pdflatex/bibtex fallback in PATH."
            )

        task_id = progress.add_task("", total=len(steps))
        for label, cmd, cwd, quiet in steps:
            progress.update(task_id, description=label)
            _run(cmd, cwd=cwd, quiet=quiet)
            progress.advance(task_id)

    report_pdf = REPORT_TEX.with_suffix(".pdf")
    dest = REPO_ROOT / "report.pdf"
    shutil.copy2(report_pdf, dest)
    _console.print(f"✓ {dest.relative_to(REPO_ROOT)}")


def _build_support_pdf(
    doc_dir: Path, tex_name: str, progress: Progress, task_id: TaskID
) -> Path:
    """Compile one support document with latexmk or pdflatex."""
    stem = Path(tex_name).stem
    if shutil.which("latexmk"):
        progress.update(task_id, description=f"{stem} (latexmk)")
        _run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", tex_name],
            cwd=doc_dir,
            quiet=True,
        )
    else:
        progress.update(task_id, description=f"{stem} pass 1")
        _run(
            ["pdflatex", "-interaction=nonstopmode", tex_name], cwd=doc_dir, quiet=True
        )
        progress.update(task_id, description=f"{stem} pass 2")
        _run(
            ["pdflatex", "-interaction=nonstopmode", tex_name], cwd=doc_dir, quiet=True
        )
    return doc_dir / f"{stem}.pdf"


def _build_support_docs() -> None:
    """Build the support PDFs and copy them to the repository root."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=_console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("", total=len(SUPPORT_DOCS))
        for subdir, tex_name, output_name in SUPPORT_DOCS:
            doc_dir = DOCS_DIR / subdir
            pdf_path = _build_support_pdf(doc_dir, tex_name, progress, task_id)
            dest = REPO_ROOT / output_name
            shutil.copy2(pdf_path, dest)
            progress.advance(task_id)

    for _, _, output_name in SUPPORT_DOCS:
        _console.print(f"✓ {(REPO_ROOT / output_name).relative_to(REPO_ROOT)}")


def _extract_log_errors(log_path: Path) -> str:
    """Extract ! error lines and context from LaTeX log file."""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    extracted = []
    for i, line in enumerate(lines):
        if line.startswith("!"):
            extracted.append(line)
            if i + 1 < len(lines) and lines[i + 1].strip():
                extracted.append(lines[i + 1])
    return "\n".join(extracted) if extracted else "(no ! errors found in log)"


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build the report PDF and/or the support PDFs."
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Build only docs/report/report.pdf.",
    )
    parser.add_argument(
        "--support-only",
        action="store_true",
        help="Build only requirements.pdf, manual.pdf, and replication.pdf.",
    )
    return parser.parse_args()


def main() -> None:
    """Parse arguments and build report and/or support documents."""
    args = _parse_args()

    if args.report_only and args.support_only:
        raise SystemExit("Choose only one of --report-only or --support-only.")

    try:
        if not args.support_only:
            _build_report()

        if not args.report_only:
            _build_support_docs()
    except subprocess.CalledProcessError as exc:
        cmd_str = " ".join(exc.cmd)
        _console.print(
            f"✗ Command failed (exit {exc.returncode}): {cmd_str}", style="bold red"
        )
        log_path = REPORT_DIR / "report.log"
        if log_path.exists():
            _console.print(_extract_log_errors(log_path), markup=False, style="red")
        else:
            combined = (exc.stdout or "") + (exc.stderr or "")
            if combined.strip():
                _console.print(combined.strip(), markup=False, style="red")
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()

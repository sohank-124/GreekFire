"""Check publication inputs without importing project scripts or training models.

Uses Git's tracked and non-ignored untracked file inventory. Checks Python syntax,
local Markdown links, required archive files, and high-confidence credential forms.
Does not inspect ignored market data, run scripts, or change Git's index.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "Project.toml", "Manifest.toml", "validation.jl",
    "test_research_validity.jl", "research_report.py", "robustness_matrix.py",
    "risk_overlay_study.py", "examples/synthetic_pipeline.jl",
    "docs/PROJECT_OVERVIEW.md", "docs/DATA_AND_CREDITS.md", "docs/PREPARATION.md",
    "CHANGELOG/RESEARCH_AUDIT.md", "CHANGELOG/RESEARCH_RESULTS_2026-09-30.md",
    "CHANGELOG/figures/equity_2026-09-30.png",
    "CHANGELOG/figures/drawdown_2026-09-30.png",
    "octave/momentum_baseline_steps.csv", "octave/momentum_baseline_summary.csv",
    "tools/recover_prices.py", "tests/test_recover_prices.py",
    "tests/fixtures/wayback_prices.csv", "requirements-recovery.txt", "docs/DATA_RECOVERY.md",
)
SECRETS = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"|\bgh[pousr]_[A-Za-z0-9]{30,}\b"
    r"|\bgithub_pat_[A-Za-z0-9_]{40,}\b"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\bsk-proj-[A-Za-z0-9_-]{30,}\b"
)
LINKS = re.compile(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")


def main():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, capture_output=True, check=True,
    )
    names = set(result.stdout.decode("utf-8").rstrip("\0").split("\0"))
    errors = []
    for name in REQUIRED:
        if name not in names or not (ROOT / name).is_file():
            errors.append(f"Missing or ignored required file: {name}")
    for name in sorted(names):
        path = ROOT / name
        if not path.is_file():
            errors.append(f"Tracked file missing from worktree: {name}")
            continue
        if path.stat().st_size >= 100 * 1024 * 1024:
            errors.append(f"File exceeds ordinary GitHub upload size: {name}")
        if name.startswith(("data/", "recovered-data/", ".local-tools/")) or Path(name).name == "sp500_ticker_start_end.csv":
            errors.append(f"Private market input included in publication inventory: {name}")
        if path.suffix.lower() not in {".py", ".jl", ".md", ".toml", ".yml", ".yaml", ".txt"}:
            continue
        content = path.read_text(encoding="utf-8-sig")
        if SECRETS.search(content):
            errors.append(f"Possible credential in {name}; inspect locally (value withheld)")
        if path.suffix == ".py":
            try:
                ast.parse(content, filename=name)
            except SyntaxError as error:
                errors.append(f"Python syntax error: {name}:{error.lineno}")
        if path.suffix == ".md":
            for target in LINKS.findall(content):
                if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                    continue
                relative = unquote(target.split("#", 1)[0])
                if relative and not (path.parent / relative).exists():
                    errors.append(f"Broken local link: {name} -> {target}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Repository checks passed ({len(names)} publication files; no training or backtest).")
    print("Credential scan covers selected text formats and known patterns, not Git history.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

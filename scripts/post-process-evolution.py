#!/usr/bin/env python3
"""
Post-process the self-evolution run output and write a summary to the wiki.

Reads the evolution output directory, extracts metrics, and writes a
formatted report to ~/wiki/queries/hermes-evolution-YYYY-MM-DD.md
"""
import json
import sys
from pathlib import Path
from datetime import datetime

TODAY = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
RUN_LOG = sys.argv[2] if len(sys.argv) > 2 else f"/home/morderith/Corvus/logs/self-evolution-{TODAY}.log"

EVOLUTION_DIR = Path("/home/morderith/Corvus/self-evolution")
WIKI_DIR = Path("/home/morderith/wiki/queries")
OUTPUT_BASE = EVOLUTION_DIR / "output"

def find_latest_output(skill_name: str) -> Path | None:
    skill_dir = OUTPUT_BASE / skill_name
    if not skill_dir.exists():
        return None
    # Find the most recent timestamped subdirectory
    subdirs = sorted(skill_dir.iterdir(), key=lambda p: p.name, reverse=True)
    for d in subdirs:
        if d.is_dir():
            return d
    return None

def read_metrics(output_dir: Path) -> dict | None:
    metrics_file = output_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    return json.loads(metrics_file.read_text())

def write_wiki_summary(skill_name: str, output_dir: Path, metrics: dict | None, run_log: str):
    lines = [
        f"# Hermes Self-Evolution Report — {TODAY}",
        "",
        f"**Skill:** {skill_name}",
        f"**Run ID:** {output_dir.name}",
        f"**Status:** {'OK' if metrics else 'FAILED (see log)'}",
        "",
        "## Result",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    if metrics:
        lines.extend([
            f"| Baseline Score | {metrics.get('baseline_score', 'N/A'):.3f} |",
            f"| Evolved Score | {metrics.get('evolved_score', 'N/A'):.3f} |",
            f"| Improvement | {metrics.get('improvement', 'N/A'):+.3f} |",
            f"| Iterations | {metrics.get('iterations', 'N/A')} |",
            f"| Optimizer | {metrics.get('optimizer', 'N/A')} |",
            f"| Elapsed | {metrics.get('elapsed_seconds', 0):.1f}s |",
            f"| Constraints Passed | {metrics.get('constraints_passed', 'N/A')} |",
        ])
    else:
        # Try to extract from run log
        if Path(run_log).exists():
            log_content = Path(run_log).read_text()
            lines.append(f"| Status | FAILED — see `{run_log}` |")
        else:
            lines.append("| Status | FAILED — no output found |")

    lines.extend([
        "",
        "## Output Location",
        "",
        f"`{output_dir}/`",
        "",
    ])

    if metrics and metrics.get("improvement", 0) > 0:
        lines.extend([
            "## Next Steps",
            "",
            f"- Review the diff: `diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md`",
            f"- Run tests: `cd {EVOLUTION_DIR} && pytest tests/`",
        ])

    wiki_path = WIKI_DIR / f"hermes-evolution-{TODAY}.md"
    wiki_path.write_text("\n".join(lines))
    print(f"Written: {wiki_path}")

def main():
    # Try github-code-review (the fallback skill after systematic-debugging plateau)
    for skill in ["github-code-review", "systematic-debugging"]:
        output_dir = find_latest_output(skill)
        if output_dir:
            metrics = read_metrics(output_dir)
            write_wiki_summary(skill, output_dir, metrics, RUN_LOG)
            return

    # No output found — write a failure report
    wiki_path = WIKI_DIR / f"hermes-evolution-{TODAY}.md"
    content = f"""# Hermes Self-Evolution Report — {TODAY}

**Status:** NO OUTPUT — evolution script did not produce any output directory.

Check the run log at: {RUN_LOG}

Possible causes:
1. API authentication failure
2. Skill not found
3. Pre-flight check failed
4. Script path incorrect
"""
    wiki_path.write_text(content)
    print(f"Written failure report: {wiki_path}")

if __name__ == "__main__":
    main()

"""Evolve a Hermes Agent skill using DSPy + GEPA/MIPROv2.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import json
import random
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from evolution.core.config import EvolutionConfig, get_hermes_agent_path, get_skills_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import skill_fitness_metric, LLMJudge, FitnessScore
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()


# ── Pre-flight validation ──────────────────────────────────────────────

def _pre_flight_check(skill_name: str, optimizer_model: str) -> dict:
    """Validate all prerequisites before running evolution.

    Returns a dict with 'ok' bool and 'message' str.
    Fails fast with specific errors rather than running a partial evaluation.
    """
    messages = []
    ok = True

    # (a) Skill SKILL.md exists
    try:
        skills_path = get_skills_path()
        skill_path = find_skill(skill_name, skills_path)
        if not skill_path:
            messages.append(f"FAIL: Skill '{skill_name}' not found in {skills_path}")
            ok = False
        else:
            messages.append(f"OK: Skill found at {skill_path.relative_to(skills_path)}")
    except Exception as e:
        messages.append(f"FAIL: Could not access skills path: {e}")
        ok = False

    # (b) Optimizer model API responds to a lightweight ping (fail fast)
    try:
        ping_lm = dspy.LM(optimizer_model)
        dspy.configure(lm=ping_lm)
        with dspy.context(lm=ping_lm):
            ping_lm("ping", max_tokens=2)
        messages.append(f"OK: Optimizer model '{optimizer_model}' responded to ping")
    except Exception as e:
        messages.append(f"FAIL: Optimizer model '{optimizer_model}' did not respond: {e}")
        ok = False

    return {"ok": ok, "messages": messages}


def _check_real_debugging_sessions() -> bool:
    """Check whether real debugging sessions exist in session history.

    Used to decide whether to re-evolve systematic-debugging or switch
    to github-code-review. Returns True if 3+ real debugging cases found.
    """
    try:
        from evolution.core.external_importers import build_dataset_from_external
        dataset = build_dataset_from_external(
            skill_name="systematic-debugging",
            skill_text="",
            sources=["hermes"],
            output_path=Path("/tmp/debugging_probe"),
            model="minimax/MiniMax-M2.7",
        )
        count = len(dataset.all_examples) if dataset.all_examples else 0
        console.print(f"  [dim]Probed session history: {count} debugging examples found[/dim]")
        return count >= 3
    except Exception as e:
        console.print(f"  [dim]Session probe failed ({e}), assuming none found[/dim]")
        return False


# ── MIPRO optimizer with jittered exponential backoff ───────────────────

class _MIPROv2WithBackoff:
    """MIPROv2 wrapped with jittered exponential backoff on compile() API errors.

    Scoped to optimizer model calls only — does not modify litellm globally.
    """

    def __init__(self, metric, auto="light", max_retries=3, initial_backoff=2.0):
        self._inner = dspy.MIPROv2(metric=metric, auto=auto)
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff

    def compile(self, *args, **kwargs):
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._inner.compile(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt == self._max_retries:
                    break
                jitter = random.uniform(0, 1)
                backoff = self._initial_backoff * (2 ** attempt) + jitter
                time.sleep(backoff)
        raise last_error


def _gepa_available() -> tuple[bool, str]:
    """Check whether GEPA is usable in this DSPy version.

    Returns (True, reason) if usable, (False, reason) if not.
    """
    import inspect

    if not hasattr(dspy, "GEPA"):
        return False, "dspy.GEPA class does not exist"

    sig = inspect.signature(dspy.GEPA.__init__)
    if "max_steps" not in sig.parameters:
        return False, "dspy.GEPA.__init__ does not accept max_steps"

    # Lightweight smoke test — instantiate without compiling
    try:
        dummy_metric = lambda *a, **k: 0.0  # noqa: E731
        _ = dspy.GEPA(metric=dummy_metric, max_steps=1)
        return True, "GEPA smoke test passed"
    except Exception as e:
        return False, f"GEPA smoke test failed: {e}"


# ── Main evolution function ────────────────────────────────────────────

def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
):
    """Main evolution function — orchestrates the full optimization loop."""

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    # ── 0. Pre-flight validation ───────────────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Pre-flight checks\n")

    checks = _pre_flight_check(skill_name, optimizer_model)
    gepa_message = ""

    for msg in checks["messages"]:
        if msg.startswith("OK"):
            console.print(f"  [green]✓[/green] {msg[3:]}")
        else:
            console.print(f"  [red]✗[/red] {msg[4:]}")

    if not checks["ok"]:
        console.print("\n[bold red]✗ Pre-flight failed — aborting evolution.[/bold red]")
        sys.exit(1)

    # ── 0b. Plateau check: systematic-debugging requires real eval data ─
    if skill_name == "systematic-debugging":
        console.print(f"\n[bold]Checking eval set diversity[/bold]")
        has_real = _check_real_debugging_sessions()
        if not has_real:
            console.print(
                "\n[bold yellow]⚠ systematic-debugging SKIPPED — eval set is plateaued[/bold yellow]"
            )
            console.print(
                "  synthetic data saturated at 0.5759–0.586 (zero content mutations across 3 runs).\n"
                "  No real debugging sessions found in session history.\n"
                "  Switching to: github-code-review\n"
            )
            # Recurse with github-code-review instead
            evolve(
                skill_name="github-code-review",
                iterations=iterations,
                eval_source=eval_source,
                dataset_path=dataset_path,
                optimizer_model=optimizer_model,
                eval_model=eval_model,
                hermes_repo=hermes_repo,
                run_tests=run_tests,
                dry_run=dry_run,
            )
            return
        else:
            console.print("  [green]✓[/green] Real debugging sessions found — proceeding with systematic-debugging")
    console.print()

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")

    skills_path = get_skills_path()
    skill_path = find_skill(skill_name, skills_path)
    if not skill_path:
        console.print(f"[red]✗ Skill '{skill_name}' not found in {skills_path}[/red]")
        sys.exit(1)

    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path.relative_to(skills_path)}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA/MIPRO optimization ({iterations} iterations)")
        console.print(f"  Would validate constraints and create PR")
        return

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
    elif eval_source == "sessiondb":
        save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill["raw"],
            sources=["claude-code", "copilot", "hermes"],
            output_path=save_path,
            model=eval_model,
        )
        if not dataset.all_examples:
            console.print("[red]✗ No relevant examples found from session history[/red]")
            sys.exit(1)
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
    elif eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(
            artifact_text=skill["raw"],
            artifact_type="skill",
        )
        # Save for reuse
        save_path = Path("datasets") / "skills" / skill_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
        sys.exit(1)

    # (c) Eval set must have at least 10 examples
    if len(dataset.all_examples) < 10:
        console.print(
            f"[red]✗ Eval set has only {len(dataset.all_examples)} examples (minimum: 10) — aborting.[/red]"
        )
        sys.exit(1)

    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate constraints on baseline ─────────────────────────────
    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_all(skill["body"], "skill")
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[yellow]⚠ Baseline skill has constraint violations — proceeding anyway[/yellow]")

    # ── 4. Set up DSPy + GEPA optimizer ─────────────────────────────────
    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    # Configure DSPy
    lm = dspy.LM(eval_model)
    dspy.configure(lm=lm)

    # Create the baseline skill module
    baseline_module = SkillModule(skill["body"])

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run GEPA/MIPRO optimization ─────────────────────────────────
    start_time = time.time()

    # Check GEPA availability once before attempting it
    gepa_ok, gepa_check_reason = _gepa_available()

    if gepa_ok:
        console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")
        console.print(f"  GEPA check: {gepa_check_reason}")
        optimizer = dspy.GEPA(
            metric=skill_fitness_metric,
            max_steps=iterations,
        )
        # Wrap compile() with backoff in case API overload hits mid-optimization
        last_error = None
        for attempt in range(4):  # 0..3 = 4 attempts total
            try:
                optimized_module = optimizer.compile(
                    baseline_module,
                    trainset=trainset,
                    valset=valset,
                )
                gepa_message = "Optimizer: GEPA"
                break
            except Exception as e:
                last_error = e
                if attempt == 3:
                    console.print(f"[yellow]GEPA compile failed after 4 attempts ({e}) — falling back to MIPROv2[/yellow]")
                    optimizer = _MIPROv2WithBackoff(
                        metric=skill_fitness_metric,
                        auto="light",
                    )
                    optimized_module = optimizer.compile(
                        baseline_module,
                        trainset=trainset,
                    )
                    gepa_message = "Optimizer: MIPROv2 (GEPA exhausted retries)"
                    break
                jitter = random.uniform(0, 1)
                backoff = 2.0 * (2 ** attempt) + jitter
                console.print(f"[yellow]GEPA compile attempt {attempt+1} failed ({e}) — retrying in {backoff:.1f}s[/yellow]")
                time.sleep(backoff)
    else:
        console.print(f"[yellow]GEPA unavailable ({gepa_check_reason}) — falling back to MIPROv2[/yellow]")
        console.print(f"\n[bold cyan]Running MIPROv2 optimization ({iterations} iterations)...[/bold cyan]\n")
        optimizer = _MIPROv2WithBackoff(
            metric=skill_fitness_metric,
            auto="light",
        )
        try:
            optimized_module = optimizer.compile(
                baseline_module,
                trainset=trainset,
            )
            gepa_message = "Optimizer: MIPROv2 (GEPA unavailable)"
        except Exception as e:
            console.print(f"[red]✗ MIPROv2 also failed: {e}[/red]")
            console.print("[red]✗ Evolution aborted — no optimizer available.[/red]")
            sys.exit(1)

    elapsed = time.time() - start_time
    console.print(f"\n  {gepa_message} — completed in {elapsed:.1f}s")

    # ── 6. Extract evolved skill text ────────────────────────────────────
    # The optimized module's predictor signature contains the evolved instructions
    evolved_body = optimized_module.get_skill_text()
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 7. Validate evolved skill ───────────────────────────────────────
    console.print(f"\n[bold]Validating evolved skill[/bold]")
    evolved_constraints = validator.validate_all(evolved_full, "skill", baseline_text=skill["raw"])
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved skill FAILED constraints — not deploying[/red]")
        # Still save for inspection
        output_path = Path("output") / skill_name / "evolved_FAILED.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {output_path}")
        return

    # ── 8. Evaluate on holdout set ──────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")

    holdout_examples = dataset.to_dspy_examples("holdout")

    baseline_scores = []
    evolved_scores = []
    for ex in holdout_examples:
        # Score baseline
        with dspy.context(lm=lm):
            baseline_pred = baseline_module(task_input=ex.task_input)
            baseline_score = skill_fitness_metric(ex, baseline_pred)
            baseline_scores.append(baseline_score)

            evolved_pred = optimized_module(task_input=ex.task_input)
            evolved_score = skill_fitness_metric(ex, evolved_pred)
            evolved_scores.append(evolved_score)

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # ── 9. Report results ──────────────────────────────────────────────
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    table.add_row(
        "Skill Size",
        f"{len(skill['body']):,} chars",
        f"{len(evolved_body):,} chars",
        f"{len(evolved_body) - len(skill['body']):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")

    console.print()
    console.print(table)

    # ── 10. Save output ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evolved skill
    (output_dir / "evolved_skill.md").write_text(evolved_full)

    # Save baseline for comparison
    (output_dir / "baseline_skill.md").write_text(skill["raw"])

    # Save metrics
    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
        "optimizer": gepa_message,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if improvement > 0:
        console.print(f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
        console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not improve skill (change: {improvement:+.3f})[/yellow]")
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, run_tests, dry_run):
    """Evolve a Hermes Agent skill using DSPy + GEPA/MIPROv2 optimization."""
    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()

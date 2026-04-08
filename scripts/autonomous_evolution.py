#!/usr/bin/env python3
"""
Autonomous Evolution Runner for Hermes Agent Self-Evolution.

This script runs weekly to:
1. Sync from upstream
2. Select a skill to evolve (rotation)
3. Run evolution experiment
4. Commit results to fork
5. Research context from available resources

Environment:
    EVOLUTION_MODEL: Override model (format: "provider/model" or "model|api_base")
    EVOLUTION_ROTATION: Rotation strategy (random, round_robin, priority)
    SKILLS_PATH: Override path to skills directory
"""

import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolution.core.config import (
    get_hermes_agent_path,
    get_skills_path,
    get_model_config,
    EvolutionConfig,
)


REPO_ROOT = Path(__file__).parent.parent.resolve()
ROTATION_FILE = REPO_ROOT / "evolution-reports" / "rotation_state.json"
# Priority skills for CTO Assistant evolution
# Tier 1: Core development and reasoning
PRIORITY_SKILLS = [
    # Core development - essential for code review and debugging
    "github/code-review",
    "software-development/systematic-debugging",
    "software-development/test-driven-development",
    "software-development/code-review",
    
    # Planning and execution - essential for project management
    "software-development/plan",
    "software-development/planning-and-task-breakdown",
    "software-development/writing-plans",
    "software-development/spec-driven-development",
    "software-development/incremental-implementation",
    
    # Architecture and design - essential for technical decisions
    "software-development/api-and-interface-design",
    "software-development/frontend-ui-engineering",
    "software-development/security-and-hardening",
    "software-development/performance-optimization",
    
    # Business operations - CTO/EnorthMedia focus
    "business/enorth-crm",
    "business/enorth-prospecting",
    "business/enorth-lead-enrichment",
    "enorth-management/nextstrat-scheduling",
    
    # DevOps - infrastructure management
    "devops/hermes-watchdog",
    "devops/corvus-git-hermes",
    "github/github-pr-workflow",
    
    # Research and analysis
    "research/duckduckgo-search",
    "enorth-industry-research/enorth-industry-research",
]


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    result = subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def sync_from_upstream() -> bool:
    """Sync from upstream NousResearch repo."""
    print("\n=== Syncing from upstream ===")
    
    # Fetch upstream
    code, stdout, stderr = run_cmd(["git", "fetch", "upstream"])
    if code != 0:
        print(f"Failed to fetch upstream: {stderr}")
        return False
    
    # Check for new commits
    code, stdout, stderr = run_cmd(["git", "log", "HEAD..upstream/main", "--oneline"])
    if code == 0 and stdout.strip():
        print(f"New commits from upstream:\n{stdout}")
        # Merge them
        code, stdout, stderr = run_cmd(["git", "merge", "upstream/main"])
        if code != 0:
            print(f"Failed to merge upstream: {stderr}")
            return False
        print("Merged upstream changes")
    else:
        print("Already up to date with upstream")
    
    return True


def load_rotation_state() -> dict:
    """Load the current rotation state."""
    if ROTATION_FILE.exists():
        with open(ROTATION_FILE) as f:
            return json.load(f)
    return {
        "last_skill": None,
        "last_attempt": None,
        "last_status": None,
        "attempt_count": 0,
        "success_count": 0,
        "rotation_index": 0,
    }


def save_rotation_state(state: dict):
    """Save the rotation state."""
    ROTATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ROTATION_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def discover_skills() -> List[str]:
    """Discover all available skills."""
    skills_path = get_skills_path()
    skills = []
    
    for skill_dir in skills_path.iterdir():
        if skill_dir.is_dir() and not skill_dir.name.startswith('.'):
            # Each subdirectory is a category
            category = skill_dir.name
            for skill_subdir in skill_dir.iterdir():
                if skill_subdir.is_dir() and not skill_subdir.name.startswith('.'):
                    # Check for SKILL.md
                    skill_file = skill_subdir / "SKILL.md"
                    if skill_file.exists():
                        skill_name = skill_subdir.name
                        skills.append(f"{category}/{skill_name}")
    
    return sorted(set(skills))


def select_next_skill(state: dict, strategy: str = "priority") -> str:
    """Select the next skill to evolve."""
    available_skills = discover_skills()
    
    if not available_skills:
        raise ValueError("No skills found")
    
    # Filter to priority skills that exist
    priority_available = [s for s in PRIORITY_SKILLS if s in available_skills or s.split("/")[-1] in [x.split("/")[-1] for x in available_skills]]
    
    if strategy == "priority" and priority_available:
        # Use round-robin through priority skills
        idx = state.get("rotation_index", 0) % len(priority_available)
        next_skill = priority_available[idx]
        
        # Match to actual skill path
        for s in available_skills:
            if s.endswith(next_skill) or s == next_skill:
                next_skill = s
                break
        
        return next_skill
    
    elif strategy == "random":
        return random.choice(available_skills)
    
    else:  # round_robin
        idx = state.get("rotation_index", 0) % len(available_skills)
        return available_skills[idx]


def run_evolution(skill_name: str, model: str, api_base: Optional[str]) -> Tuple[bool, dict]:
    """Run the evolution experiment for a skill."""
    print(f"\n=== Evolving skill: {skill_name} ===")
    print(f"Model: {model}")
    print(f"API Base: {api_base}")
    
    # Set environment for DSPy/LiteLLM
    env = os.environ.copy()
    
    # Handle local models via api_base
    if api_base:
        env["OPENAI_API_KEY"] = "local"
        env["OPENAI_BASE_URL"] = api_base
    
    # Run evolve_skill.py
    cmd = [
        sys.executable,
        "-m", "evolution.skills.evolve_skill",
        "--skill", skill_name,
        "--iterations", "10",
        "--eval-source", "synthetic",
        "--optimizer-model", model,
        "--eval-model", model,
    ]
    
    # Add hermes repo path
    try:
        hermes_path = get_hermes_agent_path()
        cmd.extend(["--hermes-repo", str(hermes_path)])
    except:
        pass
    
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    
    success = result.returncode == 0
    
    # Parse metrics from output directory
    metrics = {
        "skill": skill_name,
        "model": model,
        "api_base": api_base,
        "success": success,
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
        "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
    }
    
    # Try to load metrics.json if created
    output_dir = REPO_ROOT / "output" / skill_name
    if output_dir.exists():
        latest = sorted(output_dir.iterdir())[-1] if list(output_dir.iterdir()) else None
        if latest and (latest / "metrics.json").exists():
            with open(latest / "metrics.json") as f:
                metrics["evolution_metrics"] = json.load(f)
    
    return success, metrics


def commit_results(skill_name: str, success: bool, metrics: dict):
    """Commit evolution results to fork."""
    print(f"\n=== Committing results ===")
    
    # Stage changes
    run_cmd(["git", "add", "-A"])
    
    # Create commit message
    status = "success" if success else "failed"
    improvement = metrics.get("evolution_metrics", {}).get("improvement", 0)
    
    commit_msg = f"evolution: {skill_name} ({status}, improvement: {improvement:+.3f})"
    
    code, stdout, stderr = run_cmd(["git", "commit", "-m", commit_msg])
    if code != 0:
        print(f"Nothing to commit or commit failed: {stderr}")
        return False
    
    # Push to fork
    code, stdout, stderr = run_cmd(["git", "push", "origin", "main"])
    if code != 0:
        print(f"Failed to push: {stderr}")
        return False
    
    print(f"Committed and pushed: {commit_msg}")
    return True


def generate_report(state: dict, metrics: dict, output_path: Path):
    """Generate a markdown report for this evolution run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_path / f"evolution_{timestamp}.md"
    
    report = f"""# Evolution Report

**Timestamp:** {datetime.now().isoformat()}
**Skill:** {metrics.get('skill', 'unknown')}
**Model:** {metrics.get('model', 'unknown')}
**API Base:** {metrics.get('api_base', 'default')}
**Status:** {'SUCCESS' if metrics.get('success') else 'FAILED'}

## Evolution Metrics

```json
{json.dumps(metrics.get('evolution_metrics', {}), indent=2)}
```

## Rotation State

- **Rotation Index:** {state.get('rotation_index', 0)}
- **Attempt Count:** {state.get('attempt_count', 0)}
- **Success Count:** {state.get('success_count', 0)}
- **Last Skill:** {state.get('last_skill', 'none')}

## Output

{metrics.get('stdout', 'No output')}

## Errors

{metrics.get('stderr', 'No errors')}
"""
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"Report saved to: {report_path}")


def main():
    """Main evolution runner."""
    print("=" * 60)
    print("HERMES AGENT SELF-EVOLUTION")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")
    
    # Load rotation state
    state = load_rotation_state()
    print(f"Previous skill: {state.get('last_skill')}")
    print(f"Previous status: {state.get('last_status')}")
    
    # Sync from upstream
    if not sync_from_upstream():
        print("WARNING: Failed to sync from upstream, continuing anyway")
    
    # Get model config
    model, api_base = get_model_config()
    print(f"\nUsing model: {model}")
    if api_base:
        print(f"API base: {api_base}")
    
    # Select next skill
    skill_name = select_next_skill(state, strategy="priority")
    print(f"\nSelected skill: {skill_name}")
    
    # Run evolution
    success, metrics = run_evolution(skill_name, model, api_base)
    
    # Update rotation state
    state["last_skill"] = skill_name
    state["last_attempt"] = datetime.now().isoformat()
    state["last_status"] = "success" if success else "failed"
    state["attempt_count"] = state.get("attempt_count", 0) + 1
    if success:
        state["success_count"] = state.get("success_count", 0) + 1
    state["rotation_index"] = state.get("rotation_index", 0) + 1
    
    # Save state
    save_rotation_state(state)
    
    # Generate report
    generate_report(state, metrics, REPO_ROOT / "evaluation-reports")
    
    # Commit results
    commit_results(skill_name, success, metrics)
    
    print("\n" + "=" * 60)
    print(f"EVOLUTION {'COMPLETE' if success else 'FAILED'}")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
"""Configuration and hermes-agent repo discovery."""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    # hermes-agent repo path
    hermes_agent_path: Path = field(default_factory=lambda: get_hermes_agent_path())

    # Optimization parameters
    iterations: int = 10
    population_size: int = 5

    # LLM configuration - defaults to local, can override via env or config
    optimizer_model: str = "local/model"  # Model for GEPA reflections
    eval_model: str = "local/model"  # Model for LLM-as-judge scoring
    judge_model: str = "local/model"  # Model for dataset generation
    
    # API base URL for local/custom models
    api_base: Optional[str] = None  # e.g., "http://localhost:8080/v1"

    # Constraints
    max_skill_size: int = 15_000  # 15KB default
    max_tool_desc_size: int = 500  # chars
    max_param_desc_size: int = 200  # chars
    max_prompt_growth: float = 0.2  # 20% max growth over baseline

    # Eval dataset
    eval_dataset_size: int = 20  # Total examples to generate
    train_ratio: float = 0.5
    val_ratio: float = 0.25
    holdout_ratio: float = 0.25

    # Benchmark gating
    run_pytest: bool = True
    run_tblite: bool = False  # Expensive — opt-in
    tblite_regression_threshold: float = 0.02  # Max 2% regression allowed

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    create_pr: bool = True


def get_hermes_agent_path() -> Path:
    """Discover the hermes-agent repo path.

    Priority:
    1. HERMES_AGENT_REPO env var
    2. ~/.hermes/hermes-agent (standard install location)
    3. ../hermes-agent (sibling directory)
    """
    env_path = os.getenv("HERMES_AGENT_REPO")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    home_path = Path.home() / ".hermes" / "hermes-agent"
    if home_path.exists():
        return home_path

    sibling_path = Path(__file__).parent.parent.parent / "hermes-agent"
    if sibling_path.exists():
        return sibling_path

    raise FileNotFoundError(
        "Cannot find hermes-agent repo. Set HERMES_AGENT_REPO env var "
        "or ensure it exists at ~/.hermes/hermes-agent"
    )


def get_hermes_config() -> Dict[str, Any]:
    """Load hermes config.yaml if available."""
    config_path = Path.home() / ".hermes" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def get_model_config() -> tuple[str, Optional[str]]:
    """Get model name and API base from hermes config.
    
    Returns:
        (model_name, api_base_url) tuple
        
    Priority:
    1. EVOLUTION_MODEL env var (format: "provider/model" or "model|api_base")
    2. Hermes config main provider
    3. Local llama-server fallback
    """
    # Check env var override
    env_model = os.getenv("EVOLUTION_MODEL")
    if env_model:
        if "|" in env_model:
            model, base = env_model.split("|", 1)
            return model, base
        return env_model, None
    
    # Try hermes config
    config = get_hermes_config()
    if config:
        # Use main model as default
        main = config.get("main", {})
        if main.get("model") and main.get("base_url"):
            provider = main.get("provider", "openai")
            model = main["model"]
            # Normalize model name for DSPy/LiteLLM
            if not "/" in model:
                model = f"{provider}/{model}"
            return model, main["base_url"]
    
    # Fallback to local llama-server
    return "openai/model", "http://localhost:8080/v1"


def get_skills_path() -> Path:
    """Get the path to skills directory.
    
    Priority:
    1. SKILLS_PATH env var
    2. ~/.hermes/skills (active skills - user additions)
    3. ~/.hermes/hermes-agent/skills (bundled skills)
    """
    # Check env var override
    env_path = os.getenv("SKILLS_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p
    
    # Priority: active skills directory (user's additions)
    active_path = Path.home() / ".hermes" / "skills"
    if active_path.exists():
        return active_path
    
    # Fallback: bundled skills with hermes-agent
    bundled_path = Path.home() / ".hermes" / "hermes-agent" / "skills"
    if bundled_path.exists():
        return bundled_path
    
    raise FileNotFoundError(
        "Cannot find skills directory. Set SKILLS_PATH env var "
        "or ensure ~/.hermes/skills exists"
    )

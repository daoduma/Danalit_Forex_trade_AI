"""PPO training for the exit agent — OPTIONAL; requires stable-baselines3.

Entries must come from walk-forward TRAINING folds only. Evaluation (the only
thing that matters) replays TEST folds twice — rule manager vs RL manager on
identical entries and costs — and writes an explicit ADOPT / KEEP-RULES
verdict. Shipping 'KEEP RULES' is a fully successful outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from danalit.config import load_config
from danalit.logging_setup import setup_logging
from danalit.models.rl_exit.env import ExitEnv
from danalit.timeutil import utc_now

log = setup_logging("rl_exit")


def train_ppo(
    bars: pd.DataFrame,
    entries: list[dict],
    total_timesteps: int = 50_000,
    seed: int = 7,
    version: Optional[str] = None,
    out_base: Optional[Path] = None,
    eval_entries: Optional[list[dict]] = None,
) -> Path:
    """Train and save a PPO exit policy. Raises a clear error without SB3."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError as e:
        raise RuntimeError(
            "stable-baselines3 (and torch) are not installed. The RL exit agent is "
            "OPTIONAL — the rule-based trade manager is the default and the system is "
            "complete without this. Install with: pip install stable-baselines3 torch"
        ) from e

    env = ExitEnv(bars, entries)
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                policy_kwargs={"net_arch": [64, 64]},
                n_steps=512, batch_size=128, learning_rate=3e-4)
    callbacks = []
    if eval_entries:
        callbacks.append(EvalCallback(ExitEnv(bars, eval_entries), n_eval_episodes=20,
                                      eval_freq=5000, verbose=0))
    model.learn(total_timesteps=total_timesteps, callback=callbacks or None)

    version = version or f"rl{utc_now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = (out_base or load_config().settings.paths.absolute("models_store")) \
        / "rl_exit" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "policy.zip"
    model.save(str(path))
    (out_dir / "meta.json").write_text(json.dumps({
        "version": version, "timesteps": total_timesteps, "seed": seed,
        "n_entries": len(entries),
    }), encoding="utf-8")
    log.info("saved RL exit policy %s", path)
    return path

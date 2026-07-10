"""Train the RL refiner (TD3 or SAC) on the multi-graph QAOA refinement env.

CPU-only, office Xeon. Saves the model to models/rl_refiner_<algo>.zip.

Usage:
    python scripts/train_rl.py --algo td3 --timesteps 20000 --config configs/quick.yaml
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback

from grace_qaoa.rl.multigraph_env import MultiGraphRefineEnv

ALGOS = {"td3": TD3, "sac": SAC}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGOS), default="td3")
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--config", default="configs/quick.yaml")
    ap.add_argument("--out", default="models")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the latest autosave checkpoint if present.")
    ap.add_argument("--save-every", type=int, default=2000,
                    help="Save an autosave checkpoint every N timesteps.")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    env = MultiGraphRefineEnv(
        families=cfg["families"],
        n_nodes=cfg["n_nodes"],
        p=cfg["qaoa_p"],
        weighted=cfg.get("weighted", False),
        pool_size=cfg.get("rl_pool_size", 40),
        seed=cfg.get("seed", 0),
    )

    out_dir = Path(args.out)
    ckpt_dir = out_dir / f"ckpt_{args.algo}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"rl_refiner_{args.algo}"

    # --- resume logic: load the newest autosave if asked and available ---
    model = None
    reset_steps = True
    if args.resume:
        saves = sorted(ckpt_dir.glob("autosave_*_steps.zip"),
                       key=lambda p: int(p.stem.split("_")[1]))
        if saves:
            latest = saves[-1]
            done_steps = int(latest.stem.split("_")[1])
            print(f"Resuming from {latest.name} ({done_steps} steps done).")
            model = ALGOS[args.algo].load(latest, env=env, device="cpu")
            reset_steps = False
        else:
            print("No autosave found; starting fresh.")

    if model is None:
        model = ALGOS[args.algo](
            "MlpPolicy", env,
            verbose=1,
            device="cpu",                  # Xeon, no GPU
            learning_rate=3e-4,
            buffer_size=50_000,
            batch_size=128,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[128, 128]),
            seed=cfg.get("seed", 0),
        )

    # Periodic autosave: survives a power cut, lets --resume continue.
    autosave = CheckpointCallback(
        save_freq=args.save_every, save_path=str(ckpt_dir),
        name_prefix="autosave")

    # progress_bar needs rich+tqdm; fall back gracefully if not installed.
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True,
                    callback=autosave, reset_num_timesteps=reset_steps)
    except ImportError:
        print("(install 'rich' for a progress bar; continuing without it)")
        model.learn(total_timesteps=args.timesteps, progress_bar=False,
                    callback=autosave, reset_num_timesteps=reset_steps)

    model.save(final_path)
    print(f"Saved trained policy -> {final_path}.zip")


if __name__ == "__main__":
    main()

import os
import csv
import time
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import robosuite as suite

from robomimic.utils.file_utils import policy_from_checkpoint

os.environ["MUJOCO_GL"] = "egl"

CHECKPOINTS = {
    "BC-50": "/home/maria/projects/bc-lift-checkpoints/bc_50/model_epoch_50_fixed.pth",
    "BC-500": "/home/maria/projects/bc-lift-checkpoints/bc_500/model_epoch_500_fixed.pth",
}


def wilson(successes, total):
    z = 1.96
    p = successes / total

    denom = 1 + z**2 / total

    centre = (p + z**2 / (2 * total)) / denom

    margin = (
        z
        * math.sqrt(
            (p * (1 - p) + z**2 / (4 * total))
            / total
        )
        / denom
    )

    return centre - margin, centre + margin


def make_env():

    return suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20,
        horizon=400,
    )


def evaluate(model_name, checkpoint, episodes, device):

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    policy, _ = policy_from_checkpoint(
        ckpt_path=checkpoint,
        device=device,
        verbose=False,
    )

    successes = 0
    latencies = []

    rows = []

    for ep in range(episodes):

        np.random.seed(42 + ep)
        torch.manual_seed(42 + ep)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42 + ep)

        env = make_env()

        obs = env.reset()

        policy.start_episode()

        success = False

        steps = 400

        ep_latency = []

        for step in range(400):

            policy_obs = {
                "robot0_eef_pos": obs["robot0_eef_pos"],
                "robot0_eef_quat": obs["robot0_eef_quat"],
                "robot0_gripper_qpos": obs["robot0_gripper_qpos"],
                "object": obs["object-state"],
            }

            if device == "cuda":
                torch.cuda.synchronize()

            t1 = time.perf_counter()

            action = policy(policy_obs)

            if device == "cuda":
                torch.cuda.synchronize()

            latency = (time.perf_counter() - t1) * 1000

            ep_latency.append(latency)

            latencies.append(latency)

            obs, reward, done, info = env.step(action)

            if env._check_success():

                success = True
                steps = step + 1

                break

            if done:

                steps = step + 1
                break

        env.close()

        if success:
            successes += 1

        rows.append({
    "model": model_name,
    "episode": ep + 1,
    "seed": 42 + ep,
    "success": int(success),
    "steps": steps,
    "latency_ms": np.median(ep_latency),
})

        print(
            f"{model_name} | Episode {ep+1}/{episodes} | "
            f"{'SUCCESS' if success else 'FAIL'} | {steps} steps"
        )

    low, high = wilson(successes, episodes)

    summary = {
        "model": model_name,
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "wilson_low": low,
        "wilson_high": high,
        "median_latency": np.median(latencies),
        "mean_latency": np.mean(latencies),
    }

    return summary, rows


def save_csv(path, rows):

    with open(path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()

        writer.writerows(rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=30)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--output-dir",
        default="results/bc",
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    episode_rows = []

    for name, ckpt in CHECKPOINTS.items():

        summary, rows = evaluate(
            name,
            ckpt,
            args.episodes,
            args.device,
        )

        summaries.append(summary)

        episode_rows.extend(rows)

    save_csv(
        Path(args.output_dir) / "summary.csv",
        summaries,
    )

    save_csv(
        Path(args.output_dir) / "episodes.csv",
        episode_rows,
    )

    print("\n")
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    for s in summaries:

        print(
            f"{s['model']} : "
            f"{s['successes']}/{s['episodes']} "
            f"({100*s['success_rate']:.1f}%) "
            f"CI [{100*s['wilson_low']:.1f}%, {100*s['wilson_high']:.1f}%] "
            f"Median latency {s['median_latency']:.3f} ms"
        )


if __name__ == "__main__":
    main()

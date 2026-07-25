import argparse
import csv
import math
import os
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import robosuite as suite
import torch
from robomimic.utils.file_utils import policy_from_checkpoint


CHECKPOINTS = {
    "Square-BC-50": (
        "/home/maria/robotics_eval/square_bc_trained_models/"
        "square_bc_50/20260723200556/models/model_epoch_50.pth"
    ),
    "Square-BC-500": (
        "/home/maria/robotics_eval/square_bc_trained_models/"
        "square_bc_500/20260723201138/models/model_epoch_500.pth"
    ),
}

BASE_SEED = 42
HORIZON = 400


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")

    z = 1.96
    p = successes / total
    denominator = 1.0 + z**2 / total

    centre = (
        p + z**2 / (2.0 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            p * (1.0 - p) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )

    return (
        max(0.0, centre - margin),
        min(1.0, centre + margin),
    )


def make_env(seed: int):
    return suite.make(
        env_name="NutAssemblySquare",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20,
        horizon=HORIZON,
        ignore_done=True,
        lite_physics=False,
        hard_reset=True,
        seed=seed,
    )


def build_policy_observation(obs: dict) -> dict:
    """
    Reconstruct the robosuite 1.4.1 Square object observation:

    [nut position (3),
     nut quaternion (4),
     nut pose relative to EEF position (3),
     nut pose relative to EEF quaternion (4)]
    """
    import robosuite.utils.transform_utils as T

    required_keys = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "SquareNut_pos",
        "SquareNut_quat",
    ]

    missing = [
        key for key in required_keys
        if key not in obs
    ]

    if missing:
        raise KeyError(
            f"Missing observation keys: {missing}. "
            f"Available keys: {list(obs.keys())}"
        )

    nut_pos = np.asarray(
        obs["SquareNut_pos"],
        dtype=np.float64,
    )

    nut_quat = np.asarray(
        obs["SquareNut_quat"],
        dtype=np.float64,
    )

    eef_pos = np.asarray(
        obs["robot0_eef_pos"],
        dtype=np.float64,
    )

    eef_quat = np.asarray(
        obs["robot0_eef_quat"],
        dtype=np.float64,
    )

    # Reproduce robosuite's original relative-pose computation.
    nut_pose_world = T.pose2mat(
        (nut_pos, nut_quat)
    )

    world_pose_in_gripper = T.pose_inv(
        T.pose2mat(
            (eef_pos, eef_quat)
        )
    )

    nut_pose_in_gripper = T.pose_in_A_to_pose_in_B(
        nut_pose_world,
        world_pose_in_gripper,
    )

    relative_pos, relative_quat = T.mat2pose(
        nut_pose_in_gripper
    )

    object_vector = np.concatenate(
        [
            nut_pos,
            nut_quat,
            relative_pos,
            relative_quat,
        ],
        axis=0,
    ).astype(np.float32)

    if object_vector.shape != (14,):
        raise ValueError(
            f"Expected object vector shape (14,), "
            f"received {object_vector.shape}"
        )

    if not np.all(np.isfinite(object_vector)):
        raise ValueError(
            "Corrected object vector contains non-finite values."
        )

    return {
        "robot0_eef_pos": np.asarray(
            obs["robot0_eef_pos"],
            dtype=np.float32,
        ),
        "robot0_eef_quat": np.asarray(
            obs["robot0_eef_quat"],
            dtype=np.float32,
        ),
        "robot0_gripper_qpos": np.asarray(
            obs["robot0_gripper_qpos"],
            dtype=np.float32,
        ),
        "object": object_vector,
    }


def evaluate(
    model_name: str,
    checkpoint: str,
    episodes: int,
    device: str,
):
    print("\n" + "=" * 72)
    print(model_name)
    print("Checkpoint:", checkpoint)
    print("=" * 72)

    policy, _ = policy_from_checkpoint(
        ckpt_path=checkpoint,
        device=device,
        verbose=False,
    )

    successes = 0
    all_latencies = []
    episode_rows = []

    for episode_index in range(episodes):
        seed = BASE_SEED + episode_index

        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = make_env(seed)

        try:
            obs = env.reset()
            policy.start_episode()

            success = False
            steps = HORIZON
            episode_latencies = []

            for step in range(HORIZON):
                policy_obs = build_policy_observation(obs)

                if device == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                action = policy(policy_obs)

                if device == "cuda":
                    torch.cuda.synchronize()

                latency_ms = (
                    time.perf_counter() - start
                ) * 1000.0

                episode_latencies.append(latency_ms)
                all_latencies.append(latency_ms)

                action = np.asarray(
                    action,
                    dtype=np.float64,
                )

                if action.shape != (7,):
                    raise ValueError(
                        f"Policy returned action shape "
                        f"{action.shape}; expected (7,)"
                    )

                if not np.all(np.isfinite(action)):
                    raise ValueError(
                        "Policy returned non-finite action."
                    )

                action = np.clip(
                    action,
                    -1.0,
                    1.0,
                )

                obs, reward, done, info = env.step(action)

                if env._check_success():
                    success = True
                    steps = step + 1
                    break

                if done:
                    steps = step + 1
                    break

            if success:
                successes += 1

            row = {
                "model": model_name,
                "task": "NutAssemblySquare",
                "episode": episode_index + 1,
                "seed": seed,
                "success": int(success),
                "steps": steps,
                "median_latency_ms": float(
                    np.median(episode_latencies)
                ),
                "mean_latency_ms": float(
                    np.mean(episode_latencies)
                ),
            }

            episode_rows.append(row)

            print(
                f"{model_name} | "
                f"Episode {episode_index + 1}/{episodes} | "
                f"Seed {seed} | "
                f"{'SUCCESS' if success else 'FAIL'} | "
                f"{steps} steps | "
                f"median latency="
                f"{row['median_latency_ms']:.3f} ms"
            )

        finally:
            env.close()

    low, high = wilson(
        successes,
        episodes,
    )

    summary = {
        "model": model_name,
        "task": "NutAssemblySquare",
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "wilson_low": low,
        "wilson_high": high,
        "median_latency_ms": float(
            np.median(all_latencies)
        ),
        "mean_latency_ms": float(
            np.mean(all_latencies)
        ),
        "mean_episode_steps": float(
            np.mean(
                [row["steps"] for row in episode_rows]
            )
        ),
    }

    return summary, episode_rows


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bc_square_pilot"),
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError(
            "--episodes must be greater than zero."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []
    all_episode_rows = []

    for model_name, checkpoint in CHECKPOINTS.items():
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )

        summary, rows = evaluate(
            model_name=model_name,
            checkpoint=checkpoint,
            episodes=args.episodes,
            device=args.device,
        )

        summaries.append(summary)
        all_episode_rows.extend(rows)

    save_csv(
        args.output_dir / "summary.csv",
        summaries,
    )

    save_csv(
        args.output_dir / "episodes.csv",
        all_episode_rows,
    )

    print("\n" + "=" * 80)
    print("SQUARE PILOT RESULTS")
    print("=" * 80)

    for summary in summaries:
        print(
            f"{summary['model']}: "
            f"{summary['successes']}/"
            f"{summary['episodes']} "
            f"({summary['success_rate'] * 100:.1f}%) | "
            f"CI "
            f"[{summary['wilson_low'] * 100:.1f}%, "
            f"{summary['wilson_high'] * 100:.1f}%] | "
            f"median latency "
            f"{summary['median_latency_ms']:.3f} ms"
        )

    print("\nSaved to:")
    print(args.output_dir)


if __name__ == "__main__":
    main()

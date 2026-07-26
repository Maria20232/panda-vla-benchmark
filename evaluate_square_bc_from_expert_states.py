import argparse
import csv
import os
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import h5py
import numpy as np
import torch
import robosuite as suite
from robosuite import load_composite_controller_config
from robomimic.utils.file_utils import policy_from_checkpoint

from benchmark_bc_square_corrected import build_policy_observation


DATASET = (
    "/home/maria/robotics_eval/datasets/"
    "square/ph/low_dim_v141.hdf5"
)

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

DEMO_NAME = "demo_0"
DEFAULT_START_STEPS = [0, 20, 40, 60, 80, 100, 120]


def make_env(horizon: int):
    return suite.make(
        env_name="NutAssemblySquare",
        robots="Panda",
        controller_configs=load_composite_controller_config(
            controller="BASIC"
        ),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=horizon,
        ignore_done=True,
        lite_physics=False,
        hard_reset=True,
    )


def state_to_array(state) -> np.ndarray:
    if hasattr(state, "flatten"):
        return np.asarray(
            state.flatten(),
            dtype=np.float64,
        )

    return np.asarray(
        state,
        dtype=np.float64,
    ).reshape(-1)


def run_from_expert_state(
    policy,
    saved_states: np.ndarray,
    expert_start_step: int,
    rollout_horizon: int,
    device: str,
) -> tuple[dict, list[dict]]:
    env = make_env(rollout_horizon)

    trajectory_rows = []

    try:
        env.reset()

        current_state_length = len(
            state_to_array(env.sim.get_state())
        )

        if current_state_length != saved_states.shape[1]:
            raise RuntimeError(
                "Simulator and dataset state dimensions differ: "
                f"{current_state_length} vs {saved_states.shape[1]}"
            )

        env.sim.set_state_from_flattened(
            saved_states[expert_start_step]
        )
        env.sim.forward()

        restored_state = state_to_array(
            env.sim.get_state()
        )

        restoration_error = float(
            np.max(
                np.abs(
                    restored_state
                    - saved_states[expert_start_step]
                )
            )
        )

        obs = env._get_observations(
            force_update=True
        )

        initial_nut_pos = np.asarray(
            obs["SquareNut_pos"],
            dtype=np.float64,
        ).copy()

        initial_eef_pos = np.asarray(
            obs["robot0_eef_pos"],
            dtype=np.float64,
        ).copy()

        policy.start_episode()

        success = bool(env._check_success())
        first_success_rollout_step = 0 if success else None

        minimum_distance = float(
            np.linalg.norm(
                initial_eef_pos - initial_nut_pos
            )
        )

        maximum_reward = float("-inf")
        maximum_nut_height = float(initial_nut_pos[2])
        latencies_ms = []

        print("\n" + "=" * 80)
        print(
            f"Starting BC rollout from expert step "
            f"{expert_start_step}"
        )
        print("=" * 80)
        print("Restoration max error:", restoration_error)
        print("Initial EEF:", initial_eef_pos)
        print("Initial nut:", initial_nut_pos)
        print("Initially successful:", success)

        for rollout_step in range(rollout_horizon):
            if success:
                break

            policy_obs = build_policy_observation(obs)

            if device == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            action = np.asarray(
                policy(policy_obs),
                dtype=np.float64,
            )

            if device == "cuda":
                torch.cuda.synchronize()

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

            if action.shape != (7,):
                raise ValueError(
                    f"Expected action shape (7,), got {action.shape}"
                )

            if not np.all(np.isfinite(action)):
                raise ValueError(
                    "Policy produced a non-finite action."
                )

            action = np.clip(action, -1.0, 1.0)
            latencies_ms.append(latency_ms)

            obs, reward, done, info = env.step(action)

            eef_pos = np.asarray(
                obs["robot0_eef_pos"],
                dtype=np.float64,
            )

            nut_pos = np.asarray(
                obs["SquareNut_pos"],
                dtype=np.float64,
            )

            distance = float(
                np.linalg.norm(eef_pos - nut_pos)
            )

            success = bool(env._check_success())

            if success and first_success_rollout_step is None:
                first_success_rollout_step = rollout_step + 1

            minimum_distance = min(
                minimum_distance,
                distance,
            )

            maximum_reward = max(
                maximum_reward,
                float(reward),
            )

            maximum_nut_height = max(
                maximum_nut_height,
                float(nut_pos[2]),
            )

            trajectory_rows.append(
                {
                    "expert_start_step": expert_start_step,
                    "rollout_step": rollout_step + 1,
                    "success": int(success),
                    "reward": float(reward),
                    "eef_nut_distance": distance,
                    "nut_x": float(nut_pos[0]),
                    "nut_y": float(nut_pos[1]),
                    "nut_z": float(nut_pos[2]),
                    "eef_x": float(eef_pos[0]),
                    "eef_y": float(eef_pos[1]),
                    "eef_z": float(eef_pos[2]),
                    "action_dx": float(action[0]),
                    "action_dy": float(action[1]),
                    "action_dz": float(action[2]),
                    "action_rx": float(action[3]),
                    "action_ry": float(action[4]),
                    "action_rz": float(action[5]),
                    "action_gripper": float(action[6]),
                    "latency_ms": latency_ms,
                }
            )

            if (
                rollout_step == 0
                or (rollout_step + 1) % 50 == 0
                or success
            ):
                print(
                    f"Rollout step {rollout_step + 1:3d}/"
                    f"{rollout_horizon} | "
                    f"success={success} | "
                    f"reward={reward:.4f} | "
                    f"distance={distance:.4f} m | "
                    f"nut_z={nut_pos[2]:.4f}"
                )

        final_nut_pos = np.asarray(
            obs["SquareNut_pos"],
            dtype=np.float64,
        )

        final_eef_pos = np.asarray(
            obs["robot0_eef_pos"],
            dtype=np.float64,
        )

        executed_steps = len(trajectory_rows)

        if trajectory_rows:
            rows_for_this_start = [
                row
                for row in trajectory_rows
                if row["expert_start_step"]
                == expert_start_step
            ]
            executed_steps = len(rows_for_this_start)

        summary = {
            "model": "Square-BC-500",
            "demo": DEMO_NAME,
            "expert_start_step": expert_start_step,
            "rollout_horizon": rollout_horizon,
            "executed_rollout_steps": executed_steps,
            "success": int(success),
            "first_success_rollout_step": (
                first_success_rollout_step
                if first_success_rollout_step is not None
                else ""
            ),
            "restoration_max_error": restoration_error,
            "initial_eef_nut_distance": float(
                np.linalg.norm(
                    initial_eef_pos - initial_nut_pos
                )
            ),
            "minimum_eef_nut_distance": minimum_distance,
            "maximum_reward": maximum_reward,
            "maximum_nut_lift": float(
                maximum_nut_height - initial_nut_pos[2]
            ),
            "final_nut_dx": float(
                final_nut_pos[0] - initial_nut_pos[0]
            ),
            "final_nut_dy": float(
                final_nut_pos[1] - initial_nut_pos[1]
            ),
            "final_nut_dz": float(
                final_nut_pos[2] - initial_nut_pos[2]
            ),
            "final_eef_dx": float(
                final_eef_pos[0] - initial_eef_pos[0]
            ),
            "final_eef_dy": float(
                final_eef_pos[1] - initial_eef_pos[1]
            ),
            "final_eef_dz": float(
                final_eef_pos[2] - initial_eef_pos[2]
            ),
            "median_latency_ms": (
                float(np.median(latencies_ms))
                if latencies_ms
                else float("nan")
            ),
            "mean_latency_ms": (
                float(np.mean(latencies_ms))
                if latencies_ms
                else float("nan")
            ),
        }

        print("\nResult:")
        print(summary)

        return summary, trajectory_rows

    finally:
        env.close()


def save_csv(path: Path, rows: list[dict]):
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=list(CHECKPOINTS.keys()),
        default="Square-BC-500",
    )

    parser.add_argument(
        "--start-steps",
        type=int,
        nargs="+",
        default=DEFAULT_START_STEPS,
    )

    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "square_bc500_from_expert_states"
        ),
    )

    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    with h5py.File(DATASET, "r") as file:
        saved_states = np.asarray(
            file["data"][DEMO_NAME]["states"],
            dtype=np.float64,
        )

    invalid_steps = [
        step
        for step in args.start_steps
        if step < 0 or step >= len(saved_states)
    ]

    if invalid_steps:
        raise ValueError(
            f"Invalid start steps: {invalid_steps}. "
            f"Valid range is 0 to {len(saved_states) - 1}."
        )

    checkpoint = CHECKPOINTS[args.model]

    print("Loading policy:")
    print(args.model)
    print(checkpoint)

    policy, _ = policy_from_checkpoint(
        ckpt_path=checkpoint,
        device=args.device,
        verbose=False,
    )

    output_directory = Path(args.output_dir)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []
    all_trajectory_rows = []

    for expert_start_step in args.start_steps:
        summary, trajectory_rows = run_from_expert_state(
            policy=policy,
            saved_states=saved_states,
            expert_start_step=expert_start_step,
            rollout_horizon=args.rollout_horizon,
            device=args.device,
        )

        summary["model"] = args.model
        summaries.append(summary)
        all_trajectory_rows.extend(trajectory_rows)

    save_csv(
        output_directory / "summary.csv",
        summaries,
    )

    save_csv(
        output_directory / "trajectories.csv",
        all_trajectory_rows,
    )

    print("\n" + "=" * 80)
    print("BC FROM EXPERT STATES — FINAL RESULTS")
    print("=" * 80)

    for summary in summaries:
        print(
            f"Expert step "
            f"{summary['expert_start_step']:3d} | "
            f"success={summary['success']} | "
            f"success rollout step="
            f"{summary['first_success_rollout_step']} | "
            f"minimum distance="
            f"{summary['minimum_eef_nut_distance']:.4f} m | "
            f"maximum nut lift="
            f"{summary['maximum_nut_lift']:.4f} m"
        )

    print("\nSaved:")
    print("-", output_directory / "summary.csv")
    print("-", output_directory / "trajectories.csv")


if __name__ == "__main__":
    main()

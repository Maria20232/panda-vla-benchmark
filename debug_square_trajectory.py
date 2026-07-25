import csv
import os
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import robosuite as suite
from robosuite import load_composite_controller_config
from robomimic.utils.file_utils import policy_from_checkpoint

from benchmark_bc_square_corrected import build_policy_observation


CHECKPOINT = (
    "/home/maria/robotics_eval/square_bc_trained_models/"
    "square_bc_500/20260723201138/models/model_epoch_500.pth"
)

OUTPUT_DIR = Path("results/square_trajectory_debug")
SEED = 42
HORIZON = 400


def make_env():
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
        horizon=HORIZON,
        ignore_done=True,
        lite_physics=False,
        hard_reset=True,
        seed=SEED,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    policy, _ = policy_from_checkpoint(
        ckpt_path=CHECKPOINT,
        device="cpu",
        verbose=False,
    )

    env = make_env()

    rows = []

    try:
        obs = env.reset()
        policy.start_episode()

        initial_nut_pos = np.asarray(
            obs["SquareNut_pos"],
            dtype=np.float64,
        ).copy()

        initial_eef_pos = np.asarray(
            obs["robot0_eef_pos"],
            dtype=np.float64,
        ).copy()

        print("=" * 80)
        print("SQUARE BC-500 TRAJECTORY DEBUG")
        print("=" * 80)
        print("Initial EEF:", initial_eef_pos)
        print("Initial nut:", initial_nut_pos)
        print(
            "Initial distance:",
            np.linalg.norm(initial_eef_pos - initial_nut_pos),
        )

        minimum_distance = float("inf")
        maximum_reward = float("-inf")
        maximum_nut_height = float(initial_nut_pos[2])

        close_actions = 0
        open_actions = 0

        for step in range(HORIZON):
            policy_obs = build_policy_observation(obs)

            start = time.perf_counter()
            action = policy(policy_obs)
            latency_ms = (time.perf_counter() - start) * 1000.0

            action = np.asarray(action, dtype=np.float64)

            if action.shape != (7,):
                raise ValueError(
                    f"Expected action shape (7,), got {action.shape}"
                )

            if not np.all(np.isfinite(action)):
                raise ValueError("Policy produced non-finite actions.")

            action = np.clip(action, -1.0, 1.0)

            if action[6] > 0:
                close_actions += 1
            else:
                open_actions += 1

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

            minimum_distance = min(minimum_distance, distance)
            maximum_reward = max(maximum_reward, float(reward))
            maximum_nut_height = max(
                maximum_nut_height,
                float(nut_pos[2]),
            )

            row = {
                "step": step + 1,
                "reward": float(reward),
                "success": int(success),
                "latency_ms": latency_ms,
                "eef_x": float(eef_pos[0]),
                "eef_y": float(eef_pos[1]),
                "eef_z": float(eef_pos[2]),
                "nut_x": float(nut_pos[0]),
                "nut_y": float(nut_pos[1]),
                "nut_z": float(nut_pos[2]),
                "eef_nut_distance": distance,
                "action_dx": float(action[0]),
                "action_dy": float(action[1]),
                "action_dz": float(action[2]),
                "action_rx": float(action[3]),
                "action_ry": float(action[4]),
                "action_rz": float(action[5]),
                "action_gripper": float(action[6]),
            }

            rows.append(row)

            if step == 0 or (step + 1) % 25 == 0 or success:
                print(
                    f"Step {step + 1:3d} | "
                    f"reward={reward:.4f} | "
                    f"distance={distance:.4f} m | "
                    f"nut_z={nut_pos[2]:.4f} | "
                    f"gripper={action[6]:+.3f} | "
                    f"success={success}"
                )
                print("  action:", action)
                print("  EEF:", eef_pos)
                print("  Nut:", nut_pos)

            if success:
                print("\nSUCCESS at step", step + 1)
                break

        csv_path = OUTPUT_DIR / "trajectory.csv"

        with csv_path.open(
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

        final_eef = np.asarray(
            obs["robot0_eef_pos"],
            dtype=np.float64,
        )

        final_nut = np.asarray(
            obs["SquareNut_pos"],
            dtype=np.float64,
        )

        print("\n" + "=" * 80)
        print("TRAJECTORY SUMMARY")
        print("=" * 80)
        print("Steps:", len(rows))
        print("Minimum EEF-to-nut distance:", minimum_distance)
        print("Maximum reward:", maximum_reward)
        print(
            "Maximum nut lift:",
            maximum_nut_height - initial_nut_pos[2],
        )
        print("EEF movement:", final_eef - initial_eef_pos)
        print("Nut movement:", final_nut - initial_nut_pos)
        print("Positive gripper actions:", close_actions)
        print("Negative gripper actions:", open_actions)
        print("Saved:", csv_path)

    finally:
        env.close()


if __name__ == "__main__":
    main()

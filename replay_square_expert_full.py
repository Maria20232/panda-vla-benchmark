import csv
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import h5py
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config


DATASET = (
    "/home/maria/robotics_eval/datasets/"
    "square/ph/low_dim_v141.hdf5"
)

DEMO_NAME = "demo_0"
OUTPUT_DIR = Path("results/square_expert_replay")


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
        horizon=400,
        ignore_done=True,
        lite_physics=False,
        hard_reset=True,
    )


def state_to_array(state):
    if hasattr(state, "flatten"):
        return np.asarray(
            state.flatten(),
            dtype=np.float64,
        )

    return np.asarray(
        state,
        dtype=np.float64,
    ).reshape(-1)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(DATASET, "r") as file:
        demo = file["data"][DEMO_NAME]

        saved_states = np.asarray(
            demo["states"],
            dtype=np.float64,
        )

        expert_actions = np.asarray(
            demo["actions"],
            dtype=np.float64,
        )

        saved_rewards = np.asarray(
            demo["rewards"],
            dtype=np.float64,
        )

        saved_dones = np.asarray(
            demo["dones"],
            dtype=np.int64,
        )

    print("=" * 80)
    print("FULL SQUARE EXPERT ACTION REPLAY")
    print("=" * 80)
    print("Demo:", DEMO_NAME)
    print("States:", saved_states.shape)
    print("Actions:", expert_actions.shape)

    env = make_env()
    rows = []

    try:
        env.reset()

        if len(state_to_array(env.sim.get_state())) != saved_states.shape[1]:
            raise RuntimeError(
                "Simulator state dimension does not match dataset."
            )

        env.sim.set_state_from_flattened(
            saved_states[0]
        )
        env.sim.forward()

        restored = state_to_array(
            env.sim.get_state()
        )

        restoration_error = float(
            np.max(
                np.abs(
                    restored - saved_states[0]
                )
            )
        )

        print("Initial restoration max error:", restoration_error)

        initial_nut_pos = np.asarray(
            env._get_observations(
                force_update=True
            )["SquareNut_pos"],
            dtype=np.float64,
        ).copy()

        success = False
        first_success_step = None

        for step_index, action in enumerate(expert_actions):
            obs, reward, done, info = env.step(action)

            actual_state = state_to_array(
                env.sim.get_state()
            )

            if step_index + 1 < len(saved_states):
                expected_state = saved_states[
                    step_index + 1
                ]

                difference = actual_state - expected_state

                state_max_error = float(
                    np.max(np.abs(difference))
                )

                state_mae = float(
                    np.mean(np.abs(difference))
                )

                state_rmse = float(
                    np.sqrt(
                        np.mean(difference ** 2)
                    )
                )
            else:
                state_max_error = float("nan")
                state_mae = float("nan")
                state_rmse = float("nan")

            nut_pos = np.asarray(
                obs["SquareNut_pos"],
                dtype=np.float64,
            )

            eef_pos = np.asarray(
                obs["robot0_eef_pos"],
                dtype=np.float64,
            )

            distance = float(
                np.linalg.norm(
                    eef_pos - nut_pos
                )
            )

            success = bool(
                env._check_success()
            )

            if success and first_success_step is None:
                first_success_step = step_index + 1

            row = {
                "step": step_index + 1,
                "success": int(success),
                "environment_reward": float(reward),
                "dataset_reward": float(
                    saved_rewards[step_index]
                ),
                "dataset_done": int(
                    saved_dones[step_index]
                ),
                "state_max_error": state_max_error,
                "state_mae": state_mae,
                "state_rmse": state_rmse,
                "eef_nut_distance": distance,
                "nut_x": float(nut_pos[0]),
                "nut_y": float(nut_pos[1]),
                "nut_z": float(nut_pos[2]),
                "action_dx": float(action[0]),
                "action_dy": float(action[1]),
                "action_dz": float(action[2]),
                "action_rx": float(action[3]),
                "action_ry": float(action[4]),
                "action_rz": float(action[5]),
                "action_gripper": float(action[6]),
            }

            rows.append(row)

            if (
                step_index == 0
                or (step_index + 1) % 10 == 0
                or success
            ):
                print(
                    f"Step {step_index + 1:3d}/"
                    f"{len(expert_actions)} | "
                    f"success={success} | "
                    f"reward={reward:.4f} | "
                    f"distance={distance:.4f} | "
                    f"state RMSE={state_rmse:.6f}"
                )

        csv_path = OUTPUT_DIR / "demo_0_replay.csv"

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

        final_nut_pos = np.asarray(
            obs["SquareNut_pos"],
            dtype=np.float64,
        )

        replay_rmse_values = np.asarray(
            [
                row["state_rmse"]
                for row in rows
                if np.isfinite(row["state_rmse"])
            ],
            dtype=np.float64,
        )

        print("\n" + "=" * 80)
        print("FULL REPLAY SUMMARY")
        print("=" * 80)
        print("Success:", success)
        print("First success step:", first_success_step)
        print("Executed actions:", len(rows))
        print(
            "Maximum state RMSE:",
            float(np.max(replay_rmse_values)),
        )
        print(
            "Mean state RMSE:",
            float(np.mean(replay_rmse_values)),
        )
        print(
            "Final state RMSE:",
            float(replay_rmse_values[-1]),
        )
        print(
            "Maximum nut lift:",
            float(
                np.max(
                    [row["nut_z"] for row in rows]
                )
                - initial_nut_pos[2]
            ),
        )
        print(
            "Final nut displacement:",
            final_nut_pos - initial_nut_pos,
        )
        print("Saved:", csv_path)

    finally:
        env.close()


if __name__ == "__main__":
    main()

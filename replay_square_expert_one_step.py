import os

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
        reward_shaping=False,
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


def print_error(name, actual, expected):
    difference = actual - expected

    print(f"\n{name}")
    print("-" * 72)
    print("Shape:", actual.shape)
    print("Maximum absolute error:", float(np.max(np.abs(difference))))
    print("Mean absolute error:", float(np.mean(np.abs(difference))))
    print("RMSE:", float(np.sqrt(np.mean(difference ** 2))))

    largest_indices = np.argsort(
        np.abs(difference)
    )[-10:][::-1]

    print("\nTen largest differences:")
    for index in largest_indices:
        print(
            f"index={index:2d} | "
            f"expected={expected[index]:+.9f} | "
            f"actual={actual[index]:+.9f} | "
            f"diff={difference[index]:+.9f}"
        )


def main():
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

        saved_obs = {
            key: np.asarray(
                demo["obs"][key][0],
                dtype=np.float64,
            )
            for key in [
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot0_joint_pos",
                "object",
            ]
        }

    print("=" * 80)
    print("SQUARE EXPERT ONE-STEP REPLAY")
    print("=" * 80)
    print("Demonstration:", DEMO_NAME)
    print("Saved state shape:", saved_states.shape)
    print("Action shape:", expert_actions.shape)
    print("First expert action:", expert_actions[0])

    env = make_env()

    try:
        env.reset()

        current_state = state_to_array(
            env.sim.get_state()
        )

        print("\nCurrent simulator state length:", len(current_state))
        print("Dataset state length:", saved_states.shape[1])

        if len(current_state) != saved_states.shape[1]:
            raise RuntimeError(
                "State lengths do not match. "
                f"Simulator={len(current_state)}, "
                f"dataset={saved_states.shape[1]}"
            )

        # Restore the exact first recorded simulator state.
        env.sim.set_state_from_flattened(
            saved_states[0]
        )
        env.sim.forward()

        restored_state = state_to_array(
            env.sim.get_state()
        )

        print_error(
            "STATE RESTORATION ERROR",
            restored_state,
            saved_states[0],
        )

        # Obtain observations after restoring the state.
        if hasattr(env, "_get_observations"):
            restored_obs = env._get_observations(
                force_update=True
            )

            print("\nRESTORED OBSERVATION COMPARISON")
            print("-" * 72)

            comparisons = {
                "robot0_eef_pos": restored_obs["robot0_eef_pos"],
                "robot0_eef_quat": restored_obs["robot0_eef_quat"],
                "robot0_gripper_qpos": restored_obs["robot0_gripper_qpos"],
                "robot0_joint_pos": restored_obs["robot0_joint_pos"],
            }

            for key, actual in comparisons.items():
                actual = np.asarray(
                    actual,
                    dtype=np.float64,
                )

                expected = saved_obs[key]

                print(f"\n{key}")
                print("Dataset:", expected)
                print("Restored:", actual)
                print(
                    "Maximum absolute difference:",
                    float(
                        np.max(
                            np.abs(actual - expected)
                        )
                    ),
                )

        # Execute exactly one recorded expert action.
        env.step(expert_actions[0])

        simulated_next_state = state_to_array(
            env.sim.get_state()
        )

        print_error(
            "ONE-STEP REPLAY ERROR",
            simulated_next_state,
            saved_states[1],
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()

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

OBSERVATION_KEYS = {
    "robot0_eef_pos": "robot0_eef_pos",
    "robot0_eef_quat": "robot0_eef_quat",
    "robot0_gripper_qpos": "robot0_gripper_qpos",
    "object": "object-state",
}


def collect_dataset_statistics():
    collected = {
        key: []
        for key in OBSERVATION_KEYS
    }

    with h5py.File(DATASET, "r") as file:
        for demo_name in file["data"].keys():
            observation_group = file["data"][demo_name]["obs"]

            for dataset_key in OBSERVATION_KEYS:
                collected[dataset_key].append(
                    np.asarray(
                        observation_group[dataset_key],
                        dtype=np.float64,
                    )
                )

    statistics = {}

    for key, arrays in collected.items():
        values = np.concatenate(arrays, axis=0)

        statistics[key] = {
            "mean": np.mean(values, axis=0),
            "std": np.std(values, axis=0),
            "min": np.min(values, axis=0),
            "max": np.max(values, axis=0),
        }

    return statistics


def make_environment(seed):
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
        seed=seed,
    )


def main():
    statistics = collect_dataset_statistics()

    print("=" * 80)
    print("SQUARE OBSERVATION DISTRIBUTION CHECK")
    print("=" * 80)

    for seed in [42, 43, 44]:
        env = make_environment(seed)

        try:
            observation = env.reset()

            print(f"\nSEED {seed}")
            print("-" * 80)

            for dataset_key, env_key in OBSERVATION_KEYS.items():
                current = np.asarray(
                    observation[env_key],
                    dtype=np.float64,
                )

                mean = statistics[dataset_key]["mean"]
                std = statistics[dataset_key]["std"]

                safe_std = np.where(
                    std > 1e-8,
                    std,
                    1.0,
                )

                z_score = (
                    current - mean
                ) / safe_std

                print(f"\n{dataset_key}")
                print("Current:", current)
                print("Dataset mean:", mean)
                print("Dataset std:", std)
                print("Absolute z-scores:", np.abs(z_score))
                print(
                    "Maximum absolute z-score:",
                    float(np.max(np.abs(z_score))),
                )

        finally:
            env.close()


if __name__ == "__main__":
    main()

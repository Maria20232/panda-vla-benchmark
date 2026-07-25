import os

os.environ["MUJOCO_GL"] = "egl"

import h5py
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config

from benchmark_bc_square_corrected import build_policy_observation


DATASET = (
    "/home/maria/robotics_eval/datasets/"
    "square/ph/low_dim_v141.hdf5"
)


def main():
    dataset_objects = []

    with h5py.File(DATASET, "r") as file:
        for demo_name in file["data"].keys():
            dataset_objects.append(
                np.asarray(
                    file["data"][demo_name]["obs"]["object"],
                    dtype=np.float64,
                )
            )

    dataset_objects = np.concatenate(
        dataset_objects,
        axis=0,
    )

    mean = dataset_objects.mean(axis=0)
    std = dataset_objects.std(axis=0)
    safe_std = np.where(std > 1e-8, std, 1.0)

    for seed in [42, 43, 44]:
        env = suite.make(
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

        try:
            obs = env.reset()
            corrected = build_policy_observation(obs)["object"]

            z_scores = np.abs(
                (corrected - mean) / safe_std
            )

            print("\nSeed:", seed)
            print("Corrected object:", corrected)
            print(
                "Maximum absolute z-score:",
                float(z_scores.max()),
            )

        finally:
            env.close()


if __name__ == "__main__":
    main()

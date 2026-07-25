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


def main():
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
        seed=42,
    )

    try:
        obs = env.reset()

        print("=" * 80)
        print("ROBOSUITE 1.5.2 RESET OBSERVATIONS")
        print("=" * 80)

        for key in sorted(obs.keys()):
            value = np.asarray(obs[key])

            print(f"\n{key}")
            print("shape:", value.shape)
            print("value:", value)

        print("\n" + "=" * 80)
        print("DATASET OBSERVATION KEYS")
        print("=" * 80)

        with h5py.File(DATASET, "r") as file:
            first_demo = sorted(file["data"].keys())[0]
            obs_group = file["data"][first_demo]["obs"]

            print("First demo:", first_demo)

            for key in sorted(obs_group.keys()):
                value = np.asarray(obs_group[key][0])

                print(f"\n{key}")
                print("shape:", value.shape)
                print("first value:", value)

    finally:
        env.close()


if __name__ == "__main__":
    main()

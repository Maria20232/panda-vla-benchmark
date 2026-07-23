import os
from pprint import pprint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config


def main() -> None:
    controller_config = load_composite_controller_config(
        controller="BASIC"
    )

    print("Composite controller configuration:")
    pprint(controller_config)

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=400,
        reward_shaping=False,
    )

    try:
        low, high = env.action_spec

        print("\nEnvironment action specification")
        print("Shape:", low.shape)
        print("Low:", np.asarray(low))
        print("High:", np.asarray(high))

        print("\nRobot action limits")
        robot = env.robots[0]

        if hasattr(robot, "action_limits"):
            robot_low, robot_high = robot.action_limits
            print("Robot low:", np.asarray(robot_low))
            print("Robot high:", np.asarray(robot_high))
        else:
            print("Robot has no action_limits attribute.")

        print("\nController information")

        for name, controller in robot.part_controllers.items():
            print(f"\nController component: {name}")
            print("Class:", type(controller).__name__)

            for attribute in [
                "input_min",
                "input_max",
                "output_min",
                "output_max",
                "control_dim",
            ]:
                if hasattr(controller, attribute):
                    print(
                        f"{attribute}:",
                        getattr(controller, attribute),
                    )

    finally:
        env.close()


if __name__ == "__main__":
    main()

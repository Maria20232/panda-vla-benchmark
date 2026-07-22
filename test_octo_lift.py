import os
import random
import time
from collections import deque

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "egl"

import jax
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config

from octo.model.octo_model import OctoModel


SEED = 42
INSTRUCTION = "lift the red cube"
MAX_STEPS = 400
ACTION_CHUNK_SIZE = 4


LIFT_ACTION_STATS = {
    "mean": np.array(
        [
            0.17247589,
            0.00581450,
            -0.16882961,
            0.00308030,
            0.00513105,
            0.01149070,
            -0.40761432,
        ],
        dtype=np.float32,
    ),
    "std": np.array(
        [
            0.25917605,
            0.12982461,
            0.49394242,
            0.02227463,
            0.06348723,
            0.08342674,
            0.91315419,
        ],
        dtype=np.float32,
    ),
    "mask": np.array(
        [True, True, True, True, True, True, False],
        dtype=bool,
    ),
}


def prepare_image(obs: dict) -> np.ndarray:
    """Convert robosuite camera image to Octo format."""
    image = np.asarray(obs["agentview_image"], dtype=np.uint8)

    # Correct MuJoCo camera orientation.
    return np.flipud(image).copy()


def prepare_action(raw_action: np.ndarray) -> np.ndarray:
    """Convert one Octo output into a safe robosuite action."""
    action = np.asarray(raw_action, dtype=np.float64).copy()

    if action.shape != (7,):
        raise ValueError(f"Expected action shape (7,), received {action.shape}")

    if not np.all(np.isfinite(action)):
        raise ValueError("Octo produced NaN or infinite action values.")

    # Robosuite controller expects values in approximately [-1, 1].
    action[:6] = np.clip(action[:6], -1.0, 1.0)

    # Convert gripper prediction to an open/close command.
    action[6] = 1.0 if action[6] >= 0.0 else -1.0

    return action


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading Octo-Small...")
    model = OctoModel.load_pretrained(
        "hf://rail-berkeley/octo-small-1.5"
    )

    print("Creating Lift environment...")
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=load_composite_controller_config(
            controller="BASIC"
        ),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        horizon=MAX_STEPS,
        reward_shaping=False,
    )

    latencies_ms: list[float] = []
    step = 0
    success = False
    done = False

    try:
        obs = env.reset()

        first_image = prepare_image(obs)

        # Octo supports a history window of two frames.
        image_history = deque(
            [first_image, first_image],
            maxlen=2,
        )

        # First position is padding at the beginning.
        timestep_pad_mask = np.array(
            [[False, True]],
            dtype=bool,
        )

        task = model.create_tasks(
            texts=[INSTRUCTION]
        )

        rng = jax.random.PRNGKey(SEED)

        initial_eef = np.asarray(obs["robot0_eef_pos"]).copy()
        initial_cube = np.asarray(obs["cube_pos"]).copy()

        print("\nStarting one full Lift episode")
        print("Instruction:", INSTRUCTION)
        print("Initial EEF position:", initial_eef)
        print("Initial cube position:", initial_cube)
        print("=" * 70)

        inference_number = 0

        while step < MAX_STEPS and not done and not success:
            stacked_images = np.stack(
                list(image_history),
                axis=0,
            )[None, ...]

            octo_obs = {
                "image_primary": stacked_images,
                "timestep_pad_mask": timestep_pad_mask,
            }

            rng, inference_rng = jax.random.split(rng)

            inference_start = time.perf_counter()

            action_chunk = model.sample_actions(
                observations=octo_obs,
                tasks=task,
                unnormalization_statistics=LIFT_ACTION_STATS,
                timestep_pad_mask=timestep_pad_mask,
                rng=inference_rng,
            )

            # Ensure JAX computation has completed before timing ends.
            action_chunk = np.asarray(
                jax.device_get(action_chunk)
            )

            inference_latency_ms = (
                time.perf_counter() - inference_start
            ) * 1000.0

            latencies_ms.append(inference_latency_ms)
            inference_number += 1

            if action_chunk.shape != (1, 4, 7):
                raise RuntimeError(
                    "Expected Octo output shape (1, 4, 7), "
                    f"received {action_chunk.shape}"
                )

            # Execute Octo's four-action prediction sequentially.
            for chunk_index in range(ACTION_CHUNK_SIZE):
                if step >= MAX_STEPS or done or success:
                    break

                action = prepare_action(
                    action_chunk[0, chunk_index]
                )

                obs, reward, done, info = env.step(action)
                step += 1

                success = bool(env._check_success())

                new_image = prepare_image(obs)
                image_history.append(new_image)

                # After the first real step, both history positions are valid.
                timestep_pad_mask = np.array(
                    [[True, True]],
                    dtype=bool,
                )

                if (
                    step == 1
                    or step % 20 == 0
                    or success
                    or done
                ):
                    eef_position = np.asarray(
                        obs["robot0_eef_pos"]
                    )
                    cube_position = np.asarray(
                        obs["cube_pos"]
                    )

                    print(
                        f"Step {step:3d}/{MAX_STEPS} | "
                        f"reward={float(reward):.3f} | "
                        f"success={success} | "
                        f"gripper={action[6]:+.0f} | "
                        f"inference={inference_latency_ms:.2f} ms"
                    )
                    print("  EEF:", eef_position)
                    print("  Cube:", cube_position)

        final_eef = np.asarray(obs["robot0_eef_pos"]).copy()
        final_cube = np.asarray(obs["cube_pos"]).copy()

        print("\n" + "=" * 70)
        print("FINAL EPISODE RESULT")
        print("=" * 70)
        print("Success:", success)
        print("Steps:", step)
        print("Octo inference calls:", inference_number)
        print("Final EEF position:", final_eef)
        print("Final cube position:", final_cube)
        print(
            "Cube vertical movement:",
            float(final_cube[2] - initial_cube[2]),
        )

        if latencies_ms:
            print(
                "First inference latency:",
                f"{latencies_ms[0]:.3f} ms",
            )

            # The first call includes JAX compilation and should not be used
            # as steady-state policy latency.
            if len(latencies_ms) > 1:
                steady_state = np.asarray(
                    latencies_ms[1:],
                    dtype=np.float64,
                )

                print(
                    "Steady-state median latency:",
                    f"{np.median(steady_state):.3f} ms",
                )
                print(
                    "Steady-state mean latency:",
                    f"{np.mean(steady_state):.3f} ms",
                )

        if success:
            print("\nOcto successfully completed Lift.")
        else:
            print(
                "\nOcto did not complete Lift within "
                f"{MAX_STEPS} steps."
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()

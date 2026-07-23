import argparse
import csv
import math
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "egl"

import jax
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config

from octo.model.octo_model import OctoModel


MODEL_PATH = "hf://rail-berkeley/octo-small-1.5"
INSTRUCTION = "lift the red cube"

BASE_SEED = 42
MAX_STEPS = 400
CONTROL_FREQUENCY = 5

HISTORY_WINDOW = 2
PREDICTION_HORIZON = 4


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


def wilson_interval(
    successes: int,
    episodes: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if episodes <= 0:
        return float("nan"), float("nan")

    p = successes / episodes
    denominator = 1.0 + z * z / episodes

    centre = (
        p + z * z / (2.0 * episodes)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            p * (1.0 - p) / episodes
            + z * z / (4.0 * episodes * episodes)
        )
        / denominator
    )

    return (
        max(0.0, centre - margin),
        min(1.0, centre + margin),
    )


def prepare_image(
    observation: dict[str, Any],
    key: str,
) -> np.ndarray:
    image = np.asarray(
        observation[key],
        dtype=np.uint8,
    )

    # The saved inspection images confirmed that robosuite's raw
    # images are vertically inverted.
    return np.flipud(image).copy()


def prepare_action(
    raw_action: np.ndarray,
) -> np.ndarray:
    action = np.asarray(
        raw_action,
        dtype=np.float64,
    ).copy()

    if action.shape != (7,):
        raise ValueError(
            f"Expected action shape (7,), received {action.shape}"
        )

    if not np.all(np.isfinite(action)):
        raise ValueError(
            "Octo produced NaN or infinite action values."
        )

    action[:6] = np.clip(
        action[:6],
        -1.0,
        1.0,
    )

    # Convert the continuous gripper value to robosuite open/close.
    action[6] = 1.0 if action[6] >= 0.0 else -1.0

    return action


def create_environment():
    return suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=load_composite_controller_config(
            controller="BASIC"
        ),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[
            "agentview",
            "robot0_eye_in_hand",
        ],
        camera_heights=[
            256,
            128,
        ],
        camera_widths=[
            256,
            128,
        ],
        render_gpu_device_id=0,
        horizon=MAX_STEPS,
        control_freq=CONTROL_FREQUENCY,
        reward_shaping=False,
    )


def initialize_histories(
    observation: dict[str, Any],
):
    primary = prepare_image(
        observation,
        "agentview_image",
    )

    wrist = prepare_image(
        observation,
        "robot0_eye_in_hand_image",
    )

    primary_history = deque(
        [primary, primary],
        maxlen=HISTORY_WINDOW,
    )

    wrist_history = deque(
        [wrist, wrist],
        maxlen=HISTORY_WINDOW,
    )

    return primary_history, wrist_history


def update_histories(
    observation: dict[str, Any],
    primary_history,
    wrist_history,
) -> None:
    primary_history.append(
        prepare_image(
            observation,
            "agentview_image",
        )
    )

    wrist_history.append(
        prepare_image(
            observation,
            "robot0_eye_in_hand_image",
        )
    )


def predict_action_chunk(
    model: OctoModel,
    task: dict[str, Any],
    primary_history,
    wrist_history,
    timestep_pad_mask: np.ndarray,
    rng: jax.Array,
) -> tuple[np.ndarray, float]:
    octo_observation = {
        "image_primary": np.stack(
            list(primary_history),
            axis=0,
        )[None, ...],
        "image_wrist": np.stack(
            list(wrist_history),
            axis=0,
        )[None, ...],
        "timestep_pad_mask": timestep_pad_mask,
    }

    start = time.perf_counter()

    action_chunk = model.sample_actions(
        observations=octo_observation,
        tasks=task,
        unnormalization_statistics=LIFT_ACTION_STATS,
        timestep_pad_mask=timestep_pad_mask,
        rng=rng,
    )

    action_chunk = np.asarray(
        jax.device_get(action_chunk)
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000.0

    if action_chunk.shape != (1, 4, 7):
        raise RuntimeError(
            "Expected Octo output shape (1, 4, 7), "
            f"received {action_chunk.shape}"
        )

    return action_chunk[0], latency_ms


def temporal_ensemble(
    prediction_history: deque[np.ndarray],
) -> np.ndarray:
    """
    Combine overlapping action-chunk predictions.

    At control step t:
    - newest chunk contributes action index 0
    - previous chunk contributes action index 1
    - two-step-old chunk contributes action index 2
    - three-step-old chunk contributes action index 3

    The official Octo wrapper uses exp_weight=0 by default,
    corresponding to equal weighting.
    """
    candidates = []

    newest_first = list(reversed(prediction_history))

    for age, predicted_chunk in enumerate(newest_first):
        if age >= PREDICTION_HORIZON:
            break

        candidates.append(
            predicted_chunk[age]
        )

    if not candidates:
        raise RuntimeError(
            "Temporal ensemble received no action predictions."
        )

    return np.mean(
        np.stack(candidates, axis=0),
        axis=0,
    )


def run_episode(
    model: OctoModel,
    task: dict[str, Any],
    episode_number: int,
    episode_seed: int,
):
    random.seed(episode_seed)
    np.random.seed(episode_seed)

    env = create_environment()

    step = 0
    success = False
    done = False

    inference_count = 0
    latency_records = []

    prediction_history = deque(
        maxlen=PREDICTION_HORIZON,
    )

    gripper_close_steps = 0
    gripper_open_steps = 0

    try:
        observation = env.reset()

        primary_history, wrist_history = initialize_histories(
            observation
        )

        timestep_pad_mask = np.array(
            [[False, True]],
            dtype=bool,
        )

        rng = jax.random.PRNGKey(
            episode_seed
        )

        initial_cube = np.asarray(
            observation["cube_pos"],
            dtype=np.float64,
        ).copy()

        maximum_cube_height = float(
            initial_cube[2]
        )

        minimum_eef_cube_distance = float("inf")

        while (
            step < MAX_STEPS
            and not done
            and not success
        ):
            rng, inference_rng = jax.random.split(
                rng
            )

            action_chunk, latency_ms = predict_action_chunk(
                model=model,
                task=task,
                primary_history=primary_history,
                wrist_history=wrist_history,
                timestep_pad_mask=timestep_pad_mask,
                rng=inference_rng,
            )

            inference_count += 1

            latency_records.append(
                {
                    "episode": episode_number,
                    "seed": episode_seed,
                    "inference": inference_count,
                    "latency_ms": latency_ms,
                }
            )

            prediction_history.append(
                action_chunk
            )

            ensembled_action = temporal_ensemble(
                prediction_history
            )

            action = prepare_action(
                ensembled_action
            )

            if action[6] > 0:
                gripper_close_steps += 1
            else:
                gripper_open_steps += 1

            observation, reward, done, info = env.step(
                action
            )

            step += 1
            success = bool(
                env._check_success()
            )

            eef = np.asarray(
                observation["robot0_eef_pos"],
                dtype=np.float64,
            )

            cube = np.asarray(
                observation["cube_pos"],
                dtype=np.float64,
            )

            maximum_cube_height = max(
                maximum_cube_height,
                float(cube[2]),
            )

            minimum_eef_cube_distance = min(
                minimum_eef_cube_distance,
                float(np.linalg.norm(eef - cube)),
            )

            update_histories(
                observation=observation,
                primary_history=primary_history,
                wrist_history=wrist_history,
            )

            timestep_pad_mask = np.array(
                [[True, True]],
                dtype=bool,
            )

        final_cube = np.asarray(
            observation["cube_pos"],
            dtype=np.float64,
        ).copy()

        episode_latencies = np.asarray(
            [
                row["latency_ms"]
                for row in latency_records
            ],
            dtype=np.float64,
        )

        result = {
            "episode": episode_number,
            "seed": episode_seed,
            "instruction": INSTRUCTION,
            "success": int(success),
            "steps": step,
            "inference_calls": inference_count,
            "median_latency_ms": float(
                np.median(episode_latencies)
            ),
            "mean_latency_ms": float(
                np.mean(episode_latencies)
            ),
            "initial_cube_z": float(
                initial_cube[2]
            ),
            "final_cube_z": float(
                final_cube[2]
            ),
            "maximum_cube_height": (
                maximum_cube_height
            ),
            "maximum_cube_lift": float(
                maximum_cube_height
                - initial_cube[2]
            ),
            "minimum_eef_cube_distance": (
                minimum_eef_cube_distance
            ),
            "gripper_close_steps": (
                gripper_close_steps
            ),
            "gripper_open_steps": (
                gripper_open_steps
            ),
        }

        return result, latency_records

    finally:
        env.close()


def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
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
        default=5,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=BASE_SEED,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/octo_temporal5hz_pilot"
        ),
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError(
            "--episodes must be greater than zero."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Octo-Small...")
    model = OctoModel.load_pretrained(
        MODEL_PATH
    )

    task = model.create_tasks(
        texts=[INSTRUCTION]
    )

    episode_results = []
    all_latency_records = []

    print("\nStarting temporal-ensemble 5 Hz benchmark")
    print("Episodes:", args.episodes)
    print("Control frequency:", CONTROL_FREQUENCY, "Hz")
    print("Prediction horizon:", PREDICTION_HORIZON)
    print("=" * 80)

    for index in range(args.episodes):
        episode_number = index + 1
        episode_seed = (
            args.base_seed + index
        )

        result, latency_records = run_episode(
            model=model,
            task=task,
            episode_number=episode_number,
            episode_seed=episode_seed,
        )

        episode_results.append(result)
        all_latency_records.extend(
            latency_records
        )

        status = (
            "SUCCESS"
            if result["success"]
            else "FAIL"
        )

        print(
            f"Episode {episode_number:2d}/"
            f"{args.episodes} | "
            f"Seed {episode_seed} | "
            f"{status} | "
            f"steps={result['steps']} | "
            f"max lift="
            f"{result['maximum_cube_lift']:.4f} m | "
            f"median latency="
            f"{result['median_latency_ms']:.1f} ms"
        )

    successes = sum(
        row["success"]
        for row in episode_results
    )

    success_rate = (
        successes / args.episodes
    )

    wilson_low, wilson_high = wilson_interval(
        successes,
        args.episodes,
    )

    all_latencies = np.asarray(
        [
            row["latency_ms"]
            for row in all_latency_records
        ],
        dtype=np.float64,
    )

    summary = {
        "model": "Octo-Small-1.5",
        "task": "Lift",
        "integration": "temporal_ensemble_5hz",
        "instruction": INSTRUCTION,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": success_rate,
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
        "median_latency_ms": float(
            np.median(all_latencies)
        ),
        "mean_latency_ms": float(
            np.mean(all_latencies)
        ),
        "mean_episode_steps": float(
            np.mean(
                [
                    row["steps"]
                    for row in episode_results
                ]
            )
        ),
        "mean_maximum_cube_lift": float(
            np.mean(
                [
                    row["maximum_cube_lift"]
                    for row in episode_results
                ]
            )
        ),
        "mean_minimum_eef_cube_distance": float(
            np.mean(
                [
                    row["minimum_eef_cube_distance"]
                    for row in episode_results
                ]
            )
        ),
    }

    save_csv(
        args.output_dir / "episodes.csv",
        episode_results,
    )

    save_csv(
        args.output_dir / "inference_latency.csv",
        all_latency_records,
    )

    save_csv(
        args.output_dir / "summary.csv",
        [summary],
    )

    print("\n" + "=" * 80)
    print("TEMPORAL-ENSEMBLE 5 HZ RESULTS")
    print("=" * 80)

    print(
        f"Successes: {successes}/"
        f"{args.episodes}"
    )

    print(
        f"Success rate: "
        f"{success_rate * 100:.1f}%"
    )

    print(
        "Wilson 95% CI: "
        f"[{wilson_low * 100:.1f}%, "
        f"{wilson_high * 100:.1f}%]"
    )

    print(
        "Median latency: "
        f"{summary['median_latency_ms']:.2f} ms"
    )

    print(
        "Mean maximum cube lift: "
        f"{summary['mean_maximum_cube_lift']:.4f} m"
    )

    print("\nSaved to:")
    print(args.output_dir)


if __name__ == "__main__":
    main()

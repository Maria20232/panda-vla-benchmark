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

MAX_STEPS = 400
ACTION_CHUNK_SIZE = 4
BASE_SEED = 42


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
        [
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ],
        dtype=bool,
    ),
}


def wilson_interval(
    successes: int,
    episodes: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Calculate the Wilson 95% confidence interval."""
    if episodes <= 0:
        return float("nan"), float("nan")

    probability = successes / episodes
    denominator = 1.0 + (z * z / episodes)

    centre = (
        probability
        + z * z / (2.0 * episodes)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / episodes
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
    """Convert a robosuite image into Octo's expected orientation."""
    image = np.asarray(
        observation[key],
        dtype=np.uint8,
    )

    return np.flipud(image).copy()


def prepare_action(
    raw_action: np.ndarray,
) -> np.ndarray:
    """Convert one Octo prediction into a safe robosuite action."""
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

    action[6] = (
        1.0
        if action[6] >= 0.0
        else -1.0
    )

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
        reward_shaping=False,
    )


def initialize_histories(
    observation: dict[str, Any],
) -> tuple[deque[np.ndarray], deque[np.ndarray]]:
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
        maxlen=2,
    )

    wrist_history = deque(
        [wrist, wrist],
        maxlen=2,
    )

    return primary_history, wrist_history


def predict_action_chunk(
    model: OctoModel,
    task: dict[str, Any],
    primary_history: deque[np.ndarray],
    wrist_history: deque[np.ndarray],
    timestep_pad_mask: np.ndarray,
    rng: jax.Array,
) -> tuple[np.ndarray, float]:
    primary_batch = np.stack(
        list(primary_history),
        axis=0,
    )[None, ...]

    wrist_batch = np.stack(
        list(wrist_history),
        axis=0,
    )[None, ...]

    octo_observation = {
        "image_primary": primary_batch,
        "image_wrist": wrist_batch,
        "timestep_pad_mask": timestep_pad_mask,
    }

    start_time = time.perf_counter()

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
        time.perf_counter() - start_time
    ) * 1000.0

    if action_chunk.shape != (1, 4, 7):
        raise RuntimeError(
            "Expected Octo output shape (1, 4, 7), "
            f"received {action_chunk.shape}"
        )

    return action_chunk, latency_ms


def warm_up_model(
    model: OctoModel,
    task: dict[str, Any],
) -> None:
    """Compile Octo before measuring benchmark latency."""
    print("Running one warm-up inference...")

    env = create_environment()

    try:
        observation = env.reset()

        primary_history, wrist_history = initialize_histories(
            observation
        )

        timestep_pad_mask = np.array(
            [[False, True]],
            dtype=bool,
        )

        warmup_rng = jax.random.PRNGKey(0)

        _, warmup_latency = predict_action_chunk(
            model=model,
            task=task,
            primary_history=primary_history,
            wrist_history=wrist_history,
            timestep_pad_mask=timestep_pad_mask,
            rng=warmup_rng,
        )

        print(
            f"Warm-up completed in {warmup_latency:.2f} ms."
        )
        print(
            "The warm-up latency will not be included "
            "in the benchmark."
        )

    finally:
        env.close()


def run_episode(
    model: OctoModel,
    task: dict[str, Any],
    episode_number: int,
    episode_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    random.seed(episode_seed)
    np.random.seed(episode_seed)

    env = create_environment()

    success = False
    done = False
    step = 0
    inference_number = 0

    gripper_close_steps = 0
    gripper_open_steps = 0

    inference_records: list[dict[str, Any]] = []

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

        initial_eef = np.asarray(
            observation["robot0_eef_pos"],
            dtype=np.float64,
        ).copy()

        initial_cube = np.asarray(
            observation["cube_pos"],
            dtype=np.float64,
        ).copy()

        minimum_eef_cube_distance = float("inf")
        maximum_cube_height = float(initial_cube[2])

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

            inference_number += 1

            inference_records.append(
                {
                    "episode": episode_number,
                    "seed": episode_seed,
                    "inference": inference_number,
                    "latency_ms": latency_ms,
                }
            )

            for chunk_index in range(
                ACTION_CHUNK_SIZE
            ):
                if (
                    step >= MAX_STEPS
                    or done
                    or success
                ):
                    break

                action = prepare_action(
                    action_chunk[0, chunk_index]
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

                eef_position = np.asarray(
                    observation["robot0_eef_pos"],
                    dtype=np.float64,
                )

                cube_position = np.asarray(
                    observation["cube_pos"],
                    dtype=np.float64,
                )

                eef_cube_distance = float(
                    np.linalg.norm(
                        eef_position - cube_position
                    )
                )

                minimum_eef_cube_distance = min(
                    minimum_eef_cube_distance,
                    eef_cube_distance,
                )

                maximum_cube_height = max(
                    maximum_cube_height,
                    float(cube_position[2]),
                )

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

                timestep_pad_mask = np.array(
                    [[True, True]],
                    dtype=bool,
                )

        final_eef = np.asarray(
            observation["robot0_eef_pos"],
            dtype=np.float64,
        ).copy()

        final_cube = np.asarray(
            observation["cube_pos"],
            dtype=np.float64,
        ).copy()

        episode_latencies = np.asarray(
            [
                record["latency_ms"]
                for record in inference_records
            ],
            dtype=np.float64,
        )

        result = {
            "episode": episode_number,
            "seed": episode_seed,
            "instruction": INSTRUCTION,
            "success": int(success),
            "steps": step,
            "inference_calls": inference_number,
            "median_latency_ms": float(
                np.median(episode_latencies)
            ),
            "mean_latency_ms": float(
                np.mean(episode_latencies)
            ),
            "initial_eef_x": float(initial_eef[0]),
            "initial_eef_y": float(initial_eef[1]),
            "initial_eef_z": float(initial_eef[2]),
            "final_eef_x": float(final_eef[0]),
            "final_eef_y": float(final_eef[1]),
            "final_eef_z": float(final_eef[2]),
            "initial_cube_x": float(initial_cube[0]),
            "initial_cube_y": float(initial_cube[1]),
            "initial_cube_z": float(initial_cube[2]),
            "final_cube_x": float(final_cube[0]),
            "final_cube_y": float(final_cube[1]),
            "final_cube_z": float(final_cube[2]),
            "cube_vertical_change": float(
                final_cube[2] - initial_cube[2]
            ),
            "maximum_cube_height": maximum_cube_height,
            "maximum_cube_lift": float(
                maximum_cube_height - initial_cube[2]
            ),
            "minimum_eef_cube_distance": (
                minimum_eef_cube_distance
            ),
            "gripper_close_steps": gripper_close_steps,
            "gripper_open_steps": gripper_open_steps,
        }

        return result, inference_records

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
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Octo-Small-1.5 on the "
            "robosuite Panda Lift task."
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of evaluation episodes.",
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=BASE_SEED,
        help="Seed assigned to the first episode.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/octo_lift"),
        help="Directory for CSV results.",
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

    warm_up_model(
        model=model,
        task=task,
    )

    episode_results: list[dict[str, Any]] = []
    all_inference_records: list[dict[str, Any]] = []

    print("\nStarting Octo Lift benchmark")
    print("Episodes:", args.episodes)
    print("Base seed:", args.base_seed)
    print("Instruction:", INSTRUCTION)
    print("=" * 80)

    benchmark_start = time.perf_counter()

    for episode_index in range(args.episodes):
        episode_number = episode_index + 1
        episode_seed = (
            args.base_seed + episode_index
        )

        result, inference_records = run_episode(
            model=model,
            task=task,
            episode_number=episode_number,
            episode_seed=episode_seed,
        )

        episode_results.append(result)
        all_inference_records.extend(
            inference_records
        )

        status = (
            "SUCCESS"
            if result["success"]
            else "FAIL"
        )

        print(
            f"Episode {episode_number:3d}/"
            f"{args.episodes} | "
            f"Seed {episode_seed} | "
            f"{status} | "
            f"{result['steps']} steps | "
            f"median latency "
            f"{result['median_latency_ms']:.2f} ms | "
            f"max cube lift "
            f"{result['maximum_cube_lift']:.4f} m"
        )

    benchmark_duration = (
        time.perf_counter() - benchmark_start
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
            record["latency_ms"]
            for record in all_inference_records
        ],
        dtype=np.float64,
    )

    summary = {
        "model": "Octo-Small-1.5",
        "task": "Lift",
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
        "benchmark_duration_seconds": (
            benchmark_duration
        ),
    }

    episodes_path = (
        args.output_dir / "episodes.csv"
    )

    inference_path = (
        args.output_dir / "inference_latency.csv"
    )

    summary_path = (
        args.output_dir / "summary.csv"
    )

    save_csv(
        episodes_path,
        episode_results,
    )

    save_csv(
        inference_path,
        all_inference_records,
    )

    save_csv(
        summary_path,
        [summary],
    )

    print("\n" + "=" * 80)
    print("FINAL OCTO RESULTS")
    print("=" * 80)

    print(
        f"Successes: {successes}/"
        f"{args.episodes}"
    )

    print(
        f"Success rate: "
        f"{success_rate * 100.0:.1f}%"
    )

    print(
        "Wilson 95% CI: "
        f"[{wilson_low * 100.0:.1f}%, "
        f"{wilson_high * 100.0:.1f}%]"
    )

    print(
        "Median steady-state latency: "
        f"{summary['median_latency_ms']:.3f} ms"
    )

    print(
        "Mean steady-state latency: "
        f"{summary['mean_latency_ms']:.3f} ms"
    )

    print(
        "Mean maximum cube lift: "
        f"{summary['mean_maximum_cube_lift']:.4f} m"
    )

    print(
        "Total benchmark time: "
        f"{benchmark_duration / 60.0:.1f} minutes"
    )

    print("\nSaved:")
    print(episodes_path)
    print(inference_path)
    print(summary_path)


if __name__ == "__main__":
    main()

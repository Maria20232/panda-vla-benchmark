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
        [True, True, True, True, True, True, False],
        dtype=bool,
    ),
}


VARIANTS = {
    "all4": {
        "execute_horizon": 4,
        "use_wrist": True,
    },
    "first2": {
        "execute_horizon": 2,
        "use_wrist": True,
    },
    "first1": {
        "execute_horizon": 1,
        "use_wrist": True,
    },
    "primary_first1": {
        "execute_horizon": 1,
        "use_wrist": False,
    },
}


def wilson_interval(
    successes: int,
    episodes: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if episodes == 0:
        return float("nan"), float("nan")

    p = successes / episodes
    denominator = 1.0 + z * z / episodes
    centre = (p + z * z / (2.0 * episodes)) / denominator

    margin = (
        z
        * math.sqrt(
            p * (1.0 - p) / episodes
            + z * z / (4.0 * episodes * episodes)
        )
        / denominator
    )

    return max(0.0, centre - margin), min(1.0, centre + margin)


def prepare_image(
    observation: dict[str, Any],
    key: str,
) -> np.ndarray:
    image = np.asarray(observation[key], dtype=np.uint8)
    return np.flipud(image).copy()


def prepare_action(raw_action: np.ndarray) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float64).copy()

    if action.shape != (7,):
        raise ValueError(
            f"Expected action shape (7,), received {action.shape}"
        )

    if not np.all(np.isfinite(action)):
        raise ValueError("Octo produced non-finite action values.")

    action[:6] = np.clip(action[:6], -1.0, 1.0)
    action[6] = 1.0 if action[6] >= 0.0 else -1.0

    return action


def create_environment(use_wrist: bool):
    camera_names = ["agentview"]
    camera_heights = [256]
    camera_widths = [256]

    if use_wrist:
        camera_names.append("robot0_eye_in_hand")
        camera_heights.append(128)
        camera_widths.append(128)

    return suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=load_composite_controller_config(
            controller="BASIC"
        ),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        render_gpu_device_id=0,
        horizon=MAX_STEPS,
        control_freq=20,
        reward_shaping=False,
    )


def create_histories(
    observation: dict[str, Any],
    use_wrist: bool,
):
    primary = prepare_image(observation, "agentview_image")
    primary_history = deque([primary, primary], maxlen=2)

    wrist_history = None

    if use_wrist:
        wrist = prepare_image(
            observation,
            "robot0_eye_in_hand_image",
        )
        wrist_history = deque([wrist, wrist], maxlen=2)

    return primary_history, wrist_history


def create_octo_observation(
    primary_history,
    wrist_history,
    timestep_pad_mask: np.ndarray,
    use_wrist: bool,
) -> dict[str, Any]:
    observation = {
        "image_primary": np.stack(
            list(primary_history),
            axis=0,
        )[None, ...],
        "timestep_pad_mask": timestep_pad_mask,
    }

    if use_wrist:
        observation["image_wrist"] = np.stack(
            list(wrist_history),
            axis=0,
        )[None, ...]

    return observation


def update_histories(
    observation: dict[str, Any],
    primary_history,
    wrist_history,
    use_wrist: bool,
) -> None:
    primary_history.append(
        prepare_image(observation, "agentview_image")
    )

    if use_wrist:
        wrist_history.append(
            prepare_image(
                observation,
                "robot0_eye_in_hand_image",
            )
        )


def run_episode(
    model: OctoModel,
    task: dict[str, Any],
    variant_name: str,
    episode_number: int,
    episode_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = VARIANTS[variant_name]
    execute_horizon = config["execute_horizon"]
    use_wrist = config["use_wrist"]

    random.seed(episode_seed)
    np.random.seed(episode_seed)

    env = create_environment(use_wrist=use_wrist)

    step = 0
    success = False
    done = False
    inference_count = 0
    latencies: list[dict[str, Any]] = []

    try:
        observation = env.reset()

        primary_history, wrist_history = create_histories(
            observation,
            use_wrist=use_wrist,
        )

        timestep_pad_mask = np.array(
            [[False, True]],
            dtype=bool,
        )

        rng = jax.random.PRNGKey(episode_seed)

        initial_cube = np.asarray(
            observation["cube_pos"],
            dtype=np.float64,
        ).copy()

        maximum_cube_height = float(initial_cube[2])
        minimum_eef_cube_distance = float("inf")

        while step < MAX_STEPS and not done and not success:
            octo_observation = create_octo_observation(
                primary_history=primary_history,
                wrist_history=wrist_history,
                timestep_pad_mask=timestep_pad_mask,
                use_wrist=use_wrist,
            )

            rng, inference_rng = jax.random.split(rng)

            start = time.perf_counter()

            action_chunk = model.sample_actions(
                observations=octo_observation,
                tasks=task,
                unnormalization_statistics=LIFT_ACTION_STATS,
                timestep_pad_mask=timestep_pad_mask,
                rng=inference_rng,
            )

            action_chunk = np.asarray(
                jax.device_get(action_chunk)
            )

            latency_ms = (
                time.perf_counter() - start
            ) * 1000.0

            inference_count += 1

            latencies.append(
                {
                    "variant": variant_name,
                    "episode": episode_number,
                    "seed": episode_seed,
                    "inference": inference_count,
                    "latency_ms": latency_ms,
                }
            )

            if action_chunk.shape != (1, 4, 7):
                raise RuntimeError(
                    f"Unexpected action shape: {action_chunk.shape}"
                )

            for action_index in range(execute_horizon):
                if step >= MAX_STEPS or done or success:
                    break

                action = prepare_action(
                    action_chunk[0, action_index]
                )

                observation, reward, done, info = env.step(action)
                step += 1

                success = bool(env._check_success())

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
                    use_wrist=use_wrist,
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
            [row["latency_ms"] for row in latencies],
            dtype=np.float64,
        )

        result = {
            "variant": variant_name,
            "execute_horizon": execute_horizon,
            "use_wrist": int(use_wrist),
            "episode": episode_number,
            "seed": episode_seed,
            "success": int(success),
            "steps": step,
            "inference_calls": inference_count,
            "median_latency_ms": float(
                np.median(episode_latencies)
            ),
            "mean_latency_ms": float(
                np.mean(episode_latencies)
            ),
            "initial_cube_z": float(initial_cube[2]),
            "final_cube_z": float(final_cube[2]),
            "maximum_cube_height": maximum_cube_height,
            "maximum_cube_lift": float(
                maximum_cube_height - initial_cube[2]
            ),
            "minimum_eef_cube_distance": (
                minimum_eef_cube_distance
            ),
        }

        return result, latencies

    finally:
        env.close()


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        "--episodes-per-variant",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS.keys()),
        default=list(VARIANTS.keys()),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/octo_integration_ablation"),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Octo-Small...")
    model = OctoModel.load_pretrained(MODEL_PATH)

    task = model.create_tasks(
        texts=[INSTRUCTION]
    )

    all_episode_results: list[dict[str, Any]] = []
    all_latency_results: list[dict[str, Any]] = []

    for variant_name in args.variants:
        print("\n" + "=" * 80)
        print("Testing variant:", variant_name)
        print("Configuration:", VARIANTS[variant_name])
        print("=" * 80)

        for index in range(args.episodes_per_variant):
            episode_number = index + 1
            seed = BASE_SEED + index

            result, latency_rows = run_episode(
                model=model,
                task=task,
                variant_name=variant_name,
                episode_number=episode_number,
                episode_seed=seed,
            )

            all_episode_results.append(result)
            all_latency_results.extend(latency_rows)

            status = (
                "SUCCESS"
                if result["success"]
                else "FAIL"
            )

            print(
                f"{variant_name:15s} | "
                f"Episode {episode_number:2d}/"
                f"{args.episodes_per_variant} | "
                f"Seed {seed} | "
                f"{status} | "
                f"steps={result['steps']} | "
                f"max lift="
                f"{result['maximum_cube_lift']:.4f} m"
            )

    summaries = []

    for variant_name in args.variants:
        rows = [
            row
            for row in all_episode_results
            if row["variant"] == variant_name
        ]

        successes = sum(row["success"] for row in rows)
        episodes = len(rows)
        low, high = wilson_interval(successes, episodes)

        latency_values = np.asarray(
            [
                row["latency_ms"]
                for row in all_latency_results
                if row["variant"] == variant_name
            ],
            dtype=np.float64,
        )

        summaries.append(
            {
                "variant": variant_name,
                "episodes": episodes,
                "successes": successes,
                "success_rate": successes / episodes,
                "wilson_low": low,
                "wilson_high": high,
                "median_latency_ms": float(
                    np.median(latency_values)
                ),
                "mean_latency_ms": float(
                    np.mean(latency_values)
                ),
                "mean_maximum_cube_lift": float(
                    np.mean(
                        [
                            row["maximum_cube_lift"]
                            for row in rows
                        ]
                    )
                ),
                "mean_minimum_eef_cube_distance": float(
                    np.mean(
                        [
                            row["minimum_eef_cube_distance"]
                            for row in rows
                        ]
                    )
                ),
            }
        )

    save_csv(
        args.output_dir / "episodes.csv",
        all_episode_results,
    )

    save_csv(
        args.output_dir / "inference_latency.csv",
        all_latency_results,
    )

    save_csv(
        args.output_dir / "summary.csv",
        summaries,
    )

    print("\n" + "=" * 80)
    print("INTEGRATION ABLATION RESULTS")
    print("=" * 80)

    for row in summaries:
        print(
            f"{row['variant']:15s} | "
            f"{row['successes']}/{row['episodes']} | "
            f"SR={row['success_rate'] * 100:.1f}% | "
            f"median latency="
            f"{row['median_latency_ms']:.1f} ms | "
            f"mean max lift="
            f"{row['mean_maximum_cube_lift']:.4f} m"
        )

    print("\nSaved to:", args.output_dir)


if __name__ == "__main__":
    main()

import csv
import os
import random
import time
from collections import deque
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "egl"

import jax
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config

from octo.model.octo_model import OctoModel


SEED = 42
MAX_STEPS = 400
ACTION_CHUNK_SIZE = 4

PROMPTS = [
    "lift the red cube",
    "pick up the red cube",
    "raise the red cube",
    "grasp and lift the red cube",
    "lift the cube",
    "pick up the cube",
    "raise the cube",
    "grab the red cube",
    "take the red cube off the table",
    "move the red cube upward",
    "elevate the red cube",
]

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


def prepare_image(obs: dict, key: str) -> np.ndarray:
    image = np.asarray(obs[key], dtype=np.uint8)
    return np.flipud(image).copy()


def prepare_action(raw_action: np.ndarray) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float64).copy()

    if action.shape != (7,):
        raise ValueError(
            f"Expected action shape (7,), received {action.shape}"
        )

    if not np.all(np.isfinite(action)):
        raise ValueError("Octo produced invalid action values.")

    action[:6] = np.clip(action[:6], -1.0, 1.0)
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
        horizon=MAX_STEPS,
        reward_shaping=False,
    )


def run_episode(
    model: OctoModel,
    instruction: str,
    episode_seed: int,
) -> dict:
    random.seed(episode_seed)
    np.random.seed(episode_seed)

    env = create_environment()

    step = 0
    success = False
    done = False
    inference_count = 0
    latencies_ms = []

    gripper_close_steps = 0
    gripper_open_steps = 0

    try:
        obs = env.reset()

        required_keys = [
            "agentview_image",
            "robot0_eye_in_hand_image",
        ]

        for key in required_keys:
            if key not in obs:
                raise KeyError(
                    f"Required observation key is missing: {key}. "
                    f"Available image keys: "
                    f"{[k for k in obs if 'image' in k]}"
                )

        primary = prepare_image(
            obs,
            "agentview_image",
        )
        wrist = prepare_image(
            obs,
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

        timestep_pad_mask = np.array(
            [[False, True]],
            dtype=bool,
        )

        task = model.create_tasks(
            texts=[instruction]
        )

        rng = jax.random.PRNGKey(episode_seed)

        initial_cube = np.asarray(
            obs["cube_pos"],
            dtype=np.float64,
        ).copy()

        print("\n" + "=" * 76)
        print("Instruction:", instruction)
        print("Seed:", episode_seed)
        print("Initial cube:", initial_cube)
        print("=" * 76)

        while step < MAX_STEPS and not done and not success:
            primary_batch = np.stack(
                list(primary_history),
                axis=0,
            )[None, ...]

            wrist_batch = np.stack(
                list(wrist_history),
                axis=0,
            )[None, ...]

            octo_obs = {
                "image_primary": primary_batch,
                "image_wrist": wrist_batch,
                "timestep_pad_mask": timestep_pad_mask,
            }

            rng, inference_rng = jax.random.split(rng)

            start = time.perf_counter()

            action_chunk = model.sample_actions(
                observations=octo_obs,
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

            latencies_ms.append(latency_ms)
            inference_count += 1

            if action_chunk.shape != (1, 4, 7):
                raise RuntimeError(
                    "Expected output shape (1, 4, 7), "
                    f"received {action_chunk.shape}"
                )

            for chunk_index in range(ACTION_CHUNK_SIZE):
                if step >= MAX_STEPS or done or success:
                    break

                action = prepare_action(
                    action_chunk[0, chunk_index]
                )

                if action[6] > 0:
                    gripper_close_steps += 1
                else:
                    gripper_open_steps += 1

                obs, reward, done, info = env.step(action)
                step += 1

                success = bool(env._check_success())

                primary_history.append(
                    prepare_image(
                        obs,
                        "agentview_image",
                    )
                )
                wrist_history.append(
                    prepare_image(
                        obs,
                        "robot0_eye_in_hand_image",
                    )
                )

                timestep_pad_mask = np.array(
                    [[True, True]],
                    dtype=bool,
                )

                if step == 1 or step % 50 == 0 or success:
                    eef = np.asarray(obs["robot0_eef_pos"])
                    cube = np.asarray(obs["cube_pos"])

                    print(
                        f"Step {step:3d}/{MAX_STEPS} | "
                        f"success={success} | "
                        f"reward={float(reward):.3f} | "
                        f"gripper={action[6]:+.0f} | "
                        f"latency={latency_ms:.1f} ms"
                    )
                    print("  EEF :", eef)
                    print("  Cube:", cube)

        final_cube = np.asarray(
            obs["cube_pos"],
            dtype=np.float64,
        ).copy()

        if len(latencies_ms) > 1:
            steady_state = np.asarray(
                latencies_ms[1:],
                dtype=np.float64,
            )
        else:
            steady_state = np.asarray(
                latencies_ms,
                dtype=np.float64,
            )

        result = {
            "instruction": instruction,
            "seed": episode_seed,
            "success": int(success),
            "steps": step,
            "inference_calls": inference_count,
            "first_latency_ms": (
                float(latencies_ms[0])
                if latencies_ms
                else float("nan")
            ),
            "median_latency_ms": (
                float(np.median(steady_state))
                if steady_state.size
                else float("nan")
            ),
            "mean_latency_ms": (
                float(np.mean(steady_state))
                if steady_state.size
                else float("nan")
            ),
            "cube_start_z": float(initial_cube[2]),
            "cube_final_z": float(final_cube[2]),
            "cube_vertical_change": float(
                final_cube[2] - initial_cube[2]
            ),
            "gripper_close_steps": gripper_close_steps,
            "gripper_open_steps": gripper_open_steps,
        }

        print("\nEpisode result:")
        print(result)

        return result

    finally:
        env.close()


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")

    z = 1.96
    proportion = successes / total
    denominator = 1.0 + z**2 / total

    centre = (
        proportion + z**2 / (2.0 * total)
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (
                proportion * (1.0 - proportion)
                + z**2 / (4.0 * total)
            )
            / total
        )
        / denominator
    )

    return centre - margin, centre + margin


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Octo-Small on paired Lift instruction "
            "paraphrases and calculate PRR."
        )
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Number of paired seeds per instruction.",
    )

    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=None,
        help=(
            "Optional number of prompts to evaluate. "
            "Useful for a smoke test."
        ),
    )

    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="First paired evaluation seed.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/octo_prr",
        help="Directory for episode and summary CSV files.",
    )

    args = parser.parse_args()

    if args.repetitions <= 0:
        raise ValueError("--repetitions must be greater than zero.")

    selected_prompts = (
        PROMPTS[:args.prompt_limit]
        if args.prompt_limit is not None
        else PROMPTS
    )

    if not selected_prompts:
        raise ValueError("No prompts were selected.")

    paired_seeds = [
        args.seed_base + repetition
        for repetition in range(args.repetitions)
    ]

    print("Loading Octo-Small...")
    model = OctoModel.load_pretrained(
        "hf://rail-berkeley/octo-small-1.5"
    )

    output_directory = Path(args.output_dir)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    episode_output = output_directory / "episodes.csv"
    summary_output = output_directory / "summary.csv"
    prr_output = output_directory / "prr.csv"

    results = []

    print("\n" + "=" * 76)
    print("PAIRED PRR EXPERIMENT")
    print("=" * 76)
    print("Prompts:", len(selected_prompts))
    print("Repetitions per prompt:", args.repetitions)
    print("Paired seeds:", paired_seeds)
    print(
        "Total episodes:",
        len(selected_prompts) * args.repetitions,
    )

    for prompt_index, instruction in enumerate(selected_prompts):
        instruction_type = (
            "base"
            if prompt_index == 0
            else "paraphrase"
        )

        for repetition, episode_seed in enumerate(paired_seeds):
            print("\n" + "#" * 76)
            print(
                f"Prompt {prompt_index + 1}/"
                f"{len(selected_prompts)}"
            )
            print(
                f"Repetition {repetition + 1}/"
                f"{args.repetitions}"
            )
            print("Instruction type:", instruction_type)
            print("#" * 76)

            result = run_episode(
                model=model,
                instruction=instruction,
                episode_seed=episode_seed,
            )

            result["instruction_index"] = prompt_index
            result["instruction_type"] = instruction_type
            result["repetition"] = repetition + 1

            results.append(result)

    episode_fieldnames = [
        "instruction_index",
        "instruction_type",
        "instruction",
        "repetition",
        "seed",
        "success",
        "steps",
        "inference_calls",
        "first_latency_ms",
        "median_latency_ms",
        "mean_latency_ms",
        "cube_start_z",
        "cube_final_z",
        "cube_vertical_change",
        "gripper_close_steps",
        "gripper_open_steps",
    ]

    with episode_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=episode_fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    summary_rows = []

    for prompt_index, instruction in enumerate(selected_prompts):
        instruction_results = [
            result
            for result in results
            if result["instruction_index"] == prompt_index
        ]

        successes = sum(
            int(result["success"])
            for result in instruction_results
        )

        episodes = len(instruction_results)
        success_rate = successes / episodes

        wilson_low, wilson_high = wilson_interval(
            successes,
            episodes,
        )

        median_latencies = np.asarray(
            [
                result["median_latency_ms"]
                for result in instruction_results
                if np.isfinite(result["median_latency_ms"])
            ],
            dtype=np.float64,
        )

        mean_latencies = np.asarray(
            [
                result["mean_latency_ms"]
                for result in instruction_results
                if np.isfinite(result["mean_latency_ms"])
            ],
            dtype=np.float64,
        )

        vertical_changes = np.asarray(
            [
                result["cube_vertical_change"]
                for result in instruction_results
            ],
            dtype=np.float64,
        )

        summary_rows.append(
            {
                "instruction_index": prompt_index,
                "instruction_type": (
                    "base"
                    if prompt_index == 0
                    else "paraphrase"
                ),
                "instruction": instruction,
                "episodes": episodes,
                "successes": successes,
                "success_rate": success_rate,
                "wilson_low": wilson_low,
                "wilson_high": wilson_high,
                "median_latency_ms": (
                    float(np.median(median_latencies))
                    if median_latencies.size
                    else float("nan")
                ),
                "mean_latency_ms": (
                    float(np.mean(mean_latencies))
                    if mean_latencies.size
                    else float("nan")
                ),
                "mean_cube_vertical_change": float(
                    np.mean(vertical_changes)
                ),
            }
        )

    summary_fieldnames = list(summary_rows[0].keys())

    with summary_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    base_success_rate = summary_rows[0]["success_rate"]

    paraphrase_rows = summary_rows[1:]

    if (
        base_success_rate > 0.0
        and len(paraphrase_rows) > 0
    ):
        ratios = [
            row["success_rate"] / base_success_rate
            for row in paraphrase_rows
        ]

        prr = float(np.mean(ratios))
        prr_defined = True
    else:
        ratios = []
        prr = float("nan")
        prr_defined = False

    mean_paraphrase_success_rate = (
        float(
            np.mean(
                [
                    row["success_rate"]
                    for row in paraphrase_rows
                ]
            )
        )
        if paraphrase_rows
        else float("nan")
    )

    prr_rows = [
        {
            "base_instruction": summary_rows[0]["instruction"],
            "base_success_rate": base_success_rate,
            "number_of_paraphrases": len(paraphrase_rows),
            "mean_paraphrase_success_rate": (
                mean_paraphrase_success_rate
            ),
            "prr_defined": int(prr_defined),
            "prr": prr,
        }
    ]

    with prr_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(prr_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(prr_rows)

    print("\n" + "=" * 76)
    print("FINAL PAIRED PROMPT RESULTS")
    print("=" * 76)

    for row in summary_rows:
        print(
            f"{row['instruction']!r}: "
            f"{row['successes']}/{row['episodes']} "
            f"({100.0 * row['success_rate']:.1f}%) | "
            f"95% CI "
            f"[{100.0 * row['wilson_low']:.1f}%, "
            f"{100.0 * row['wilson_high']:.1f}%] | "
            f"median latency="
            f"{row['median_latency_ms']:.2f} ms"
        )

    if prr_defined:
        print(f"\nPRR: {prr:.4f}")
    else:
        print(
            "\nPRR: undefined because the base "
            "instruction success rate is zero."
        )

    print("\nSaved:")
    print("-", episode_output)
    print("-", summary_output)
    print("-", prr_output)


if __name__ == "__main__":
    main()

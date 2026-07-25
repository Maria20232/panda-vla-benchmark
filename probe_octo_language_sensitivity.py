import csv
import os
from pathlib import Path
from collections import deque

os.environ["MUJOCO_GL"] = "egl"

import jax
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config

from octo.model.octo_model import OctoModel


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

SEEDS = list(range(42, 52))
OUTPUT_DIR = Path("results/octo_language_sensitivity")

LIFT_ACTION_STATS = {
    "mean": np.array(
        [
            0.17247589,
            0.00581450,
            -0.16882961,
            0.00308030,
            0.00513105,
            0.01149070,
            0.00000000,
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


def prepare_image(obs, key):
    return np.flipud(
        np.asarray(obs[key], dtype=np.uint8)
    ).copy()


def create_environment(seed):
    np.random.seed(seed)

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
        camera_heights=[256, 128],
        camera_widths=[256, 128],
        horizon=400,
        reward_shaping=False,
        seed=seed,
    )


def cosine_similarity(first, second):
    denominator = (
        np.linalg.norm(first)
        * np.linalg.norm(second)
    )

    if denominator <= 1e-12:
        return float("nan")

    return float(
        np.dot(first, second) / denominator
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Octo-Small...")
    model = OctoModel.load_pretrained(
        "hf://rail-berkeley/octo-small-1.5"
    )

    rows = []

    for seed in SEEDS:
        env = create_environment(seed)

        try:
            obs = env.reset()

            primary = prepare_image(
                obs,
                "agentview_image",
            )

            wrist = prepare_image(
                obs,
                "robot0_eye_in_hand_image",
            )

            octo_obs = {
                "image_primary": np.stack(
                    [primary, primary],
                    axis=0,
                )[None, ...],
                "image_wrist": np.stack(
                    [wrist, wrist],
                    axis=0,
                )[None, ...],
                "timestep_pad_mask": np.array(
                    [[False, True]],
                    dtype=bool,
                ),
            }

            base_action = None

            for prompt_index, instruction in enumerate(PROMPTS):
                task = model.create_tasks(
                    texts=[instruction]
                )

                # Same random key across prompts for fair comparison.
                rng = jax.random.PRNGKey(seed)

                action_chunk = model.sample_actions(
                    observations=octo_obs,
                    tasks=task,
                    unnormalization_statistics=LIFT_ACTION_STATS,
                    timestep_pad_mask=np.array(
                        [[False, True]],
                        dtype=bool,
                    ),
                    rng=rng,
                )

                action_chunk = np.asarray(
                    jax.device_get(action_chunk),
                    dtype=np.float64,
                )[0]

                first_action = action_chunk[0]

                if base_action is None:
                    base_action = first_action.copy()

                difference = first_action - base_action

                rows.append(
                    {
                        "seed": seed,
                        "instruction_index": prompt_index,
                        "instruction_type": (
                            "base"
                            if prompt_index == 0
                            else "paraphrase"
                        ),
                        "instruction": instruction,
                        "action_l2_difference_from_base": float(
                            np.linalg.norm(difference)
                        ),
                        "action_cosine_similarity_to_base": (
                            cosine_similarity(
                                first_action,
                                base_action,
                            )
                        ),
                        "dx": float(first_action[0]),
                        "dy": float(first_action[1]),
                        "dz": float(first_action[2]),
                        "rx": float(first_action[3]),
                        "ry": float(first_action[4]),
                        "rz": float(first_action[5]),
                        "gripper": float(first_action[6]),
                    }
                )

                print(
                    f"Seed {seed} | "
                    f"{instruction!r} | "
                    f"L2 from base="
                    f"{rows[-1]['action_l2_difference_from_base']:.4f} | "
                    f"cosine="
                    f"{rows[-1]['action_cosine_similarity_to_base']:.4f}"
                )

        finally:
            env.close()

    episodes_file = OUTPUT_DIR / "action_probe.csv"

    with episodes_file.open(
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

    summary_rows = []

    for prompt_index, instruction in enumerate(PROMPTS):
        selected = [
            row
            for row in rows
            if row["instruction_index"] == prompt_index
        ]

        l2_values = np.asarray(
            [
                row["action_l2_difference_from_base"]
                for row in selected
            ],
            dtype=np.float64,
        )

        cosine_values = np.asarray(
            [
                row["action_cosine_similarity_to_base"]
                for row in selected
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
                "seeds": len(selected),
                "mean_action_l2_difference_from_base": float(
                    np.mean(l2_values)
                ),
                "median_action_l2_difference_from_base": float(
                    np.median(l2_values)
                ),
                "mean_action_cosine_similarity_to_base": float(
                    np.mean(cosine_values)
                ),
                "minimum_action_cosine_similarity_to_base": float(
                    np.min(cosine_values)
                ),
            }
        )

    summary_file = OUTPUT_DIR / "summary.csv"

    with summary_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved:")
    print("-", episodes_file)
    print("-", summary_file)


if __name__ == "__main__":
    main()

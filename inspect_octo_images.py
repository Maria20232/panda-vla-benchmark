import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
from PIL import Image
import robosuite as suite
from robosuite import load_composite_controller_config


SEED = 42
OUTPUT_DIR = Path("results/octo_image_inspection")


def summarize_image(name: str, image: np.ndarray) -> None:
    print(f"\n{name}")
    print("Shape:", image.shape)
    print("Dtype:", image.dtype)
    print("Minimum pixel:", image.min())
    print("Maximum pixel:", image.max())
    print("Mean pixel:", float(image.mean()))
    print("Finite:", bool(np.isfinite(image).all()))


def save_image(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    Image.fromarray(image).save(path)


def main() -> None:
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = suite.make(
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
        horizon=400,
        reward_shaping=False,
    )

    try:
        obs = env.reset()

        print("Available image keys:")
        for key in obs:
            if "image" in key:
                print("-", key)

        primary_raw = np.asarray(
            obs["agentview_image"]
        )

        wrist_raw = np.asarray(
            obs["robot0_eye_in_hand_image"]
        )

        # This is the correction currently used by our Octo benchmark.
        primary_flipped = np.flipud(primary_raw).copy()
        wrist_flipped = np.flipud(wrist_raw).copy()

        summarize_image(
            "Primary raw",
            primary_raw,
        )
        summarize_image(
            "Primary vertically flipped",
            primary_flipped,
        )
        summarize_image(
            "Wrist raw",
            wrist_raw,
        )
        summarize_image(
            "Wrist vertically flipped",
            wrist_flipped,
        )

        save_image(
            OUTPUT_DIR / "agentview_raw.png",
            primary_raw,
        )
        save_image(
            OUTPUT_DIR / "agentview_flipped.png",
            primary_flipped,
        )
        save_image(
            OUTPUT_DIR / "wrist_raw.png",
            wrist_raw,
        )
        save_image(
            OUTPUT_DIR / "wrist_flipped.png",
            wrist_flipped,
        )

        # This reproduces the exact image-history arrays used in our benchmark.
        primary_history = np.stack(
            [primary_flipped, primary_flipped],
            axis=0,
        )[None, ...]

        wrist_history = np.stack(
            [wrist_flipped, wrist_flipped],
            axis=0,
        )[None, ...]

        timestep_pad_mask = np.array(
            [[False, True]],
            dtype=bool,
        )

        print("\nOcto input tensors")
        print(
            "image_primary shape:",
            primary_history.shape,
        )
        print(
            "image_primary dtype:",
            primary_history.dtype,
        )
        print(
            "image_wrist shape:",
            wrist_history.shape,
        )
        print(
            "image_wrist dtype:",
            wrist_history.dtype,
        )
        print(
            "timestep_pad_mask:",
            timestep_pad_mask,
        )

        np.save(
            OUTPUT_DIR / "image_primary.npy",
            primary_history,
        )
        np.save(
            OUTPUT_DIR / "image_wrist.npy",
            wrist_history,
        )
        np.save(
            OUTPUT_DIR / "timestep_pad_mask.npy",
            timestep_pad_mask,
        )

        print("\nSaved image inspection files to:")
        print(OUTPUT_DIR)

    finally:
        env.close()


if __name__ == "__main__":
    main()

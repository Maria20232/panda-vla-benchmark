import h5py
import numpy as np
import torch

from robomimic.utils.file_utils import policy_from_checkpoint


DATASET = (
    "/home/maria/robotics_eval/datasets/"
    "square/ph/low_dim_v141.hdf5"
)

CHECKPOINTS = {
    "Square-BC-50": (
        "/home/maria/robotics_eval/square_bc_trained_models/"
        "square_bc_50/20260723200556/models/model_epoch_50.pth"
    ),
    "Square-BC-500": (
        "/home/maria/robotics_eval/square_bc_trained_models/"
        "square_bc_500/20260723201138/models/model_epoch_500.pth"
    ),
}

NUM_DEMOS = 20


def make_observation(obs_group, index):
    return {
        "object": np.asarray(
            obs_group["object"][index],
            dtype=np.float32,
        ),
        "robot0_eef_pos": np.asarray(
            obs_group["robot0_eef_pos"][index],
            dtype=np.float32,
        ),
        "robot0_eef_quat": np.asarray(
            obs_group["robot0_eef_quat"][index],
            dtype=np.float32,
        ),
        "robot0_gripper_qpos": np.asarray(
            obs_group["robot0_gripper_qpos"][index],
            dtype=np.float32,
        ),
    }


def evaluate_model(name, checkpoint):
    policy, _ = policy_from_checkpoint(
        ckpt_path=checkpoint,
        device="cpu",
        verbose=False,
    )

    expert_actions = []
    predicted_actions = []

    with h5py.File(DATASET, "r") as file:
        demo_names = sorted(
            file["data"].keys(),
            key=lambda value: int(value.split("_")[1]),
        )[:NUM_DEMOS]

        for demo_name in demo_names:
            demo = file["data"][demo_name]
            obs_group = demo["obs"]

            observation = make_observation(
                obs_group,
                index=0,
            )

            expert_action = np.asarray(
                demo["actions"][0],
                dtype=np.float32,
            )

            policy.start_episode()

            predicted_action = np.asarray(
                policy(observation),
                dtype=np.float32,
            )

            expert_actions.append(expert_action)
            predicted_actions.append(predicted_action)

            print("\n" + "-" * 72)
            print(name, "|", demo_name)
            print("Expert:   ", expert_action)
            print("Predicted:", predicted_action)
            print(
                "Absolute difference:",
                np.abs(predicted_action - expert_action),
            )

    expert_actions = np.asarray(expert_actions)
    predicted_actions = np.asarray(predicted_actions)

    error = predicted_actions - expert_actions

    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))

    dot = np.sum(
        predicted_actions * expert_actions,
        axis=1,
    )

    denominator = (
        np.linalg.norm(predicted_actions, axis=1)
        * np.linalg.norm(expert_actions, axis=1)
    )

    valid = denominator > 1e-8

    cosine = float(
        np.mean(dot[valid] / denominator[valid])
    )

    gripper_accuracy = float(
        np.mean(
            np.sign(predicted_actions[:, 6])
            == np.sign(expert_actions[:, 6])
        )
    )

    print("\n" + "=" * 80)
    print(name, "FIRST-ACTION SUMMARY")
    print("=" * 80)
    print("Demonstrations:", len(expert_actions))
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("Mean cosine similarity:", cosine)
    print("Gripper sign accuracy:", gripper_accuracy)

    print("\nPer-action MAE:")
    per_action_mae = np.mean(
        np.abs(error),
        axis=0,
    )

    for index, value in enumerate(per_action_mae):
        print(f"Action {index}: {value:.6f}")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    for name, checkpoint in CHECKPOINTS.items():
        evaluate_model(name, checkpoint)


if __name__ == "__main__":
    main()

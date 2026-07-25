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

MAX_SAMPLES = 5000


def load_samples():
    observations = []
    expert_actions = []

    with h5py.File(DATASET, "r") as file:
        for demo_name in sorted(file["data"].keys()):
            demo = file["data"][demo_name]
            obs_group = demo["obs"]
            actions = np.asarray(demo["actions"], dtype=np.float32)

            count = min(
                len(actions),
                MAX_SAMPLES - len(expert_actions),
            )

            for index in range(count):
                observations.append(
                    {
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
                )

                expert_actions.append(actions[index])

            if len(expert_actions) >= MAX_SAMPLES:
                break

    return observations, np.asarray(expert_actions)


def evaluate(name, checkpoint, observations, expert_actions):
    policy, _ = policy_from_checkpoint(
        ckpt_path=checkpoint,
        device="cpu",
        verbose=False,
    )

    predictions = []

    for observation in observations:
        policy.start_episode()

        predicted = np.asarray(
            policy(observation),
            dtype=np.float32,
        )

        predictions.append(predicted)

    predictions = np.asarray(predictions)

    errors = predictions - expert_actions

    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))

    numerator = np.sum(
        predictions * expert_actions,
        axis=1,
    )

    denominator = (
        np.linalg.norm(predictions, axis=1)
        * np.linalg.norm(expert_actions, axis=1)
    )

    valid = denominator > 1e-8

    cosine_similarity = float(
        np.mean(
            numerator[valid] / denominator[valid]
        )
    )

    gripper_sign_accuracy = float(
        np.mean(
            np.sign(predictions[:, 6])
            == np.sign(expert_actions[:, 6])
        )
    )

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print("Samples:", len(predictions))
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("Mean cosine similarity:", cosine_similarity)
    print(
        "Gripper sign accuracy:",
        gripper_sign_accuracy,
    )

    print("\nPer-action MAE:")
    for index, value in enumerate(
        np.mean(np.abs(errors), axis=0)
    ):
        print(f"Action {index}: {value:.6f}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    observations, expert_actions = load_samples()

    print("Loaded samples:", len(observations))
    print("Expert action shape:", expert_actions.shape)

    for name, checkpoint in CHECKPOINTS.items():
        evaluate(
            name,
            checkpoint,
            observations,
            expert_actions,
        )


if __name__ == "__main__":
    main()

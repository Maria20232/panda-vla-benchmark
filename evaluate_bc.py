import os
import time

import numpy as np
import robosuite as suite
import torch

from robomimic.utils.file_utils import policy_from_checkpoint

os.environ["MUJOCO_GL"] = "egl"

CHECKPOINT = (
    "/home/maria/projects/bc-lift-checkpoints/"
    "bc_50/model_epoch_50_fixed.pth"
)

NUM_EPISODES = 30
HORIZON = 400
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

policy, _ = policy_from_checkpoint(
    ckpt_path=CHECKPOINT,
    device=device,
    verbose=False,
)

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=False,
    control_freq=20,
    horizon=HORIZON,
)

successes = 0
episode_steps = []
latencies_ms = []

for episode in range(NUM_EPISODES):
    episode_seed = SEED + episode
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)

    obs = env.reset()
    policy.start_episode()
    success = False

    for step in range(HORIZON):
        policy_obs = {
            "robot0_eef_pos": obs["robot0_eef_pos"],
            "robot0_eef_quat": obs["robot0_eef_quat"],
            "robot0_gripper_qpos": obs["robot0_gripper_qpos"],
            "object": obs["object-state"],
        }

        start = time.perf_counter()
        action = policy(policy_obs)
        latencies_ms.append((time.perf_counter() - start) * 1000)

        obs, reward, done, info = env.step(action)

        if env._check_success():
            success = True
            successes += 1
            episode_steps.append(step + 1)
            break

        if done:
            episode_steps.append(step + 1)
            break
    else:
        episode_steps.append(HORIZON)

    print(
        f"Episode {episode + 1}/{NUM_EPISODES}: "
        f"{'SUCCESS' if success else 'FAIL'} "
        f"after {episode_steps[-1]} steps"
    )

success_rate = successes / NUM_EPISODES

print("\nResults")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Device: {device}")
print(f"Successes: {successes}/{NUM_EPISODES}")
print(f"Success rate: {success_rate * 100:.1f}%")
print(f"Median policy latency: {np.median(latencies_ms):.3f} ms")
print(f"Mean episode length: {np.mean(episode_steps):.1f} steps")

env.close()

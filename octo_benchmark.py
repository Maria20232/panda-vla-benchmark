import numpy as np
import jax
import time
import csv
import robosuite as suite
from octo.model.octo_model import OctoModel

# ========== CONFIGURATION ==========
TASKS = ["Lift", "Stack", "Door", "NutAssembly", "PickPlace"]
INSTRUCTIONS = {
    "Lift": {
        "orig": "lift the red cube",
        "para1": "raise the red block",
        "para2": "elevate the red cube"
    },
    "Stack": {
        "orig": "stack the red cube on the green platform",
        "para1": "place the red block onto the green stand",
        "para2": "put the red cube on top of the green platform"
    },
    "Door": {
        "orig": "open the door",
        "para1": "rotate the door handle",
        "para2": "unlatch the door"
    },
    "NutAssembly": {
        "orig": "fit the square nut onto the peg",
        "para1": "attach the nut to the peg",
        "para2": "place the square nut on the peg"
    },
    "PickPlace": {
        "orig": "pick up the red cube and place it in the bin",
        "para1": "grasp the red block and put it into the bin",
        "para2": "lift the red cube and drop it in the container"
    }
}
NUM_REPS = 10               # repetitions per instruction
MAX_STEPS = 400
SEED_BASE = 42

# ========== LOAD OCTO MODEL ==========
model = OctoModel.load_pretrained('hf://rail-berkeley/octo-small-1.5')

# ========== RUN ONE EPISODE ==========
def run_episode(task, instruction, rep):
    """Run one episode and return (success, steps, median_latency_ms)."""
    # Deterministic seed for reproducibility (using numpy's RNG)
    seed = SEED_BASE + hash((task, instruction, rep)) % 10000
    np.random.seed(seed)

    # Create environment
    env = suite.make(
        env_name=task,
        robots="Panda",
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        horizon=MAX_STEPS,
    )

    # Reset environment (no seed argument in this robosuite version)
    obs = env.reset()
    rng = jax.random.PRNGKey(seed)
    success = False
    step = 0
    latencies = []

    while step < MAX_STEPS:
        # Convert observation to Octo format
        img = obs["agentview_image"]                # (256,256,3)
        img = img[None, None]                       # (1,1,256,256,3)
        img = np.concatenate([img, img], axis=1)    # (1,2,256,256,3) – history 2
        octo_obs = {
            "image_primary": img,
            "timestep_pad_mask": np.array([[True, True]]),
        }

        # Inference with timing
        start = time.perf_counter()
        action_batch = model.sample_actions(octo_obs, tasks={}, rng=rng)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)      # milliseconds

        action = np.array(action_batch[0, 0])       # (7,)
        obs, reward, done, info = env.step(action)

        success = info.get("success", False)
        if done or success:
            break

        step += 1
        rng = jax.random.PRNGKey(seed + step)       # update RNG

    env.close()
    median_latency = np.median(latencies) if latencies else 0.0
    return success, step, median_latency

# ========== MAIN BENCHMARK LOOP ==========
csv_file = "octo_results.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["task", "instruction_type", "instruction_text", "rep", "success", "steps", "latency_ms"])

    for task in TASKS:
        for instr_type, instr_text in INSTRUCTIONS[task].items():
            for rep in range(NUM_REPS):
                print(f"Running {task} | {instr_type} | rep {rep+1}/{NUM_REPS}")
                success, steps, lat = run_episode(task, instr_text, rep)
                writer.writerow([task, instr_type, instr_text, rep, int(success), steps, lat])
                print(f"  -> success={success}, steps={steps}, latency={lat:.2f}ms")

print(f"\n✅ Results saved to {csv_file}")

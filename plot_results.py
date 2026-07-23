import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.size"] = 12

summary = pd.read_csv("results/bc100/summary.csv")

output = Path("results/bc100/figures")
output.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Success Rate
# -----------------------------

plt.figure(figsize=(6,4))

plt.bar(
    summary["model"],
    summary["success_rate"] * 100,
)

plt.ylabel("Success Rate (%)")
plt.ylim(0,100)
plt.title("Lift Task Success Rate")

for i,v in enumerate(summary["success_rate"]*100):
    plt.text(i,v+2,f"{v:.1f}%",ha="center")

plt.tight_layout()
plt.savefig(output/"success_rate.png")
plt.close()

# -----------------------------
# Latency
# -----------------------------

plt.figure(figsize=(6,4))

plt.bar(
    summary["model"],
    summary["median_latency"],
)

plt.ylabel("Median Latency (ms)")
plt.title("Policy Inference Latency")

for i,v in enumerate(summary["median_latency"]):
    plt.text(i,v+0.05,f"{v:.2f}",ha="center")

plt.tight_layout()
plt.savefig(output/"latency.png")
plt.close()

print("Figures saved to:")
print(output)

import pandas as pd
import numpy as np

df = pd.read_csv("octo_results.csv")

print("=== Overall ===")
print(f"Total episodes: {len(df)}")
print(f"Success rate: {df['success'].mean()*100:.1f}%")
print(f"Median latency: {df['latency_ms'].median():.1f} ms")
print(f"Mean latency: {df['latency_ms'].mean():.1f} ms")

print("\n=== Per Task ===")
for task in df['task'].unique():
    sub = df[df['task']==task]
    print(f"{task}: SR={sub['success'].mean()*100:.1f}%, latency={sub['latency_ms'].median():.1f}ms")

print("\n=== Per Instruction Type (Success Rate) ===")
print(df.groupby(['task','instruction_type'])['success'].mean().unstack())

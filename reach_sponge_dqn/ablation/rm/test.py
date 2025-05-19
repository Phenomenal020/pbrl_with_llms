import pandas as pd
import matplotlib.pyplot as plt

# Prepare the data - Generated from genscores.py
data = {
    "Checkpoint": [32000, 96000, 200000],
    "Without BT Model": [(46 + 48) / 2, (46 + 48) / 2, (48 + 60) / 2],
    "With BT Model": [(50 + 56) / 2, (54 + 42) / 2, (48 + 65) / 2]
}

df = pd.DataFrame(data)

# Plotting
plt.figure()
plt.plot(df["Checkpoint"], df["Without BT Model"], marker='o', label="Without BT Model", color='blue')
plt.plot(df["Checkpoint"], df["With BT Model"], marker='o', label="With BT Model", color='orange')

plt.xlabel("Checkpoint (Steps)")
plt.ylabel("Average Accuracy (%)")
plt.title("Reward Model Accuracy vs. Checkpoint")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
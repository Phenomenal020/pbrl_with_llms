import matplotlib.pyplot as plt

# Data
checkpoints = [1000, 5000, 10000]
models = ["naive_rl", "standard_rl", "enhanced_rl"]

data = {
    "naive_rl": [ [7/40, 4/37, 3/40], [27/40, 26/37, 29/40], [17.295, 24.975, 15.625] ],
    "standard_rl": [ [5/40, 3/40, 9/40], [33/40, 31/37, 25/40], [18.60, 17.375, 18.45] ],
    "enhanced_rl": [ [7/40, 5/40, 1/40], [31/40, 26/40, 28/40], [17.30, 26.375, 32.55] ]
}
metrics = ["Success Rate per episode", "Collision Rate per episode", "Mean Episode Length"]

# Create a montage: 1 row, 3 columns
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

for i, (ax, metric) in enumerate(zip(axes, metrics)):
    for model in models:
        ax.plot(checkpoints, data[model][i], marker='o', label=model)
    ax.set_xlabel("Checkpoint Steps")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.grid(True)
    if i == 0:
        ax.legend()

# Save the montage to a file
fig.savefig('comparative.png', dpi=600)

# Optionally display
plt.show()
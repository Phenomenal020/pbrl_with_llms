import matplotlib.pyplot as plt
import numpy as np

# Define the data
labels = [
    "no rag\n03-mini",
    "rag\n03-mini",
    "rag\n4o-mini",
    "no rag\n4o-mini",
    "rag\n03-mini\nH=3",
    "no rag\n03-mini\nH=3",
    "rag\n03-mini\nH=6",
    "no rag\n03-mini\nH=6",
    "rag\nno rank\n4o-mini",
    "rag\nno rank\n03-mini",
    "no rag\nno rank\n03-mini",
]

accuracy = [0.8272, 0.804, 0.8295, 0.7991, 0.8291, 0.8386, 0.7464, 0.7557, 0.6053, 0.6932, 0.7404]
invalids = [3, 1.5, 13, 13, 6, 4, 5, 2, 7, 4, 2]

x = np.arange(len(labels))
width = 0.35

# ---------- Plot 1: Balanced Accuracy ----------
fig1, ax1 = plt.subplots(figsize=(7, 5))
bars1 = ax1.bar(x, accuracy, width, color='steelblue')
ax1.set_ylabel('Balanced Accuracy')
ax1.set_title('Balanced Accuracy per Model')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha='right')
ax1.set_ylim(0.5, 0.9)

# Add text labels above bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

fig1.tight_layout()
fig1.savefig("accuracy_plot.png", dpi=300, bbox_inches='tight')


# ---------- Plot 2: Invalid Indices ----------
fig2, ax2 = plt.subplots(figsize=(7, 5))
bars2 = ax2.bar(x, invalids, width, color='darkorange')
ax2.set_ylabel('Invalid Indices')
ax2.set_title('Invalid Indices per Model')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right')

# Add text labels above bars
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.1f}", ha='center', va='bottom', fontsize=8)

fig2.tight_layout()
fig2.savefig("invalid_indices_plot.png", dpi=300, bbox_inches='tight')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Data
configurations = [
    "Rank + RAG R", "Rank + RAG NR", "Rank + RAG R, seg=3", "Rank + RAG NR, seg=3",
    "Rank + RAG R, seg=10", "Rank + RAG NR, seg=10", "Rank + No RAG R", "Rank + No RAG NR",
    "No Rank + RAG R", "No Rank + RAG NR", "No Rank + No RAG R", "No Rank + No RAG NR"
]


# Generated from ablation_reward.py
uniform_acc = [0.3406, 0.5500, 0.3188, 0.4313, 0.4099, 0.4231, 0.3591, 0.3833, 0.3333, 0.4375, 0.3345, 0.5395]
entropy_acc = [0.4062, 0.3932, 0.4551, 0.4522, 0.3826, 0.5357, 0.4328, 0.6731, 0.6905, 0.5783, 0.6213, 0.5885]

# Sort by average accuracy (descending)
data = list(zip(configurations, uniform_acc, entropy_acc))
data.sort(key=lambda x: (x[1] + x[2]) / 2, reverse=True)
configurations_sorted, uniform_sorted, entropy_sorted = zip(*data)

# Bar positions
x = range(len(configurations_sorted))
bar_width = 0.35

# Plot
fig, ax = plt.subplots(figsize=(14, 7))

ax.bar([i - bar_width/2 for i in x], uniform_sorted, width=bar_width, label='Uniform Sampling', color='skyblue')
ax.bar([i + bar_width/2 for i in x], entropy_sorted, width=bar_width, label='Entropy Sampling', color='orange')

# Annotations
for i in x:
    ax.text(i - bar_width/2, uniform_sorted[i] + 0.01, f"{uniform_sorted[i]:.2f}", ha='center', va='bottom', fontsize=10)
    ax.text(i + bar_width/2, entropy_sorted[i] + 0.01, f"{entropy_sorted[i]:.2f}", ha='center', va='bottom', fontsize=10)

# X-axis labels and title
ax.set_xticks(x)
ax.set_xticklabels(configurations_sorted, rotation=60, ha='right', fontsize=10)
ax.set_ylabel('LLM Label Accuracy', fontsize=12)
ax.set_title('LLM Label Accuracies: Uniform vs. Entropy Sampling (Sorted by Average)', fontsize=14)
ax.legend(fontsize=10)

plt.tight_layout(pad=2.0)
plt.savefig('llm_accuracies_vertical.png', dpi=600)
plt.show()

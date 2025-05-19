import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Data
configurations = [
    "Rank + RAG R", "Rank + RAG NR", "Rank + RAG R, seg=3", "Rank + RAG NR, seg=3",
    "Rank + RAG R, seg=10", "Rank + RAG NR, seg=10", "Rank + No RAG R", "Rank + No RAG NR",
    "No Rank + RAG R", "No Rank + RAG NR", "No Rank + No RAG R", "No Rank + RAG NR"
]
uniform_acc = [0.3406, 0.5500, 0.3188, 0.4313, 0.4099, 0.4231, 0.3591, 0.3833, 0.3333, 0.4375, 0.3345, 0.5395]
entropy_acc = [0.4062, 0.3932, 0.4551, 0.4522, 0.3826, 0.5357, 0.4328, 0.6731, 0.6905, 0.5783, 0.6213, 0.5885]

# Plot with larger bars and spacing
y = range(len(configurations))
fig, ax = plt.subplots(figsize=(10, 7))
bar_height = 0.6  # Increase bar thickness

# Horizontal bars
ax.barh([i + bar_height/2 for i in y], uniform_acc, height=bar_height, label='Uniform Sampling')
ax.barh([i - bar_height/2 for i in y], entropy_acc, height=bar_height, label='Entropy Sampling')

# Annotations
for i in y:
    ax.text(uniform_acc[i] + 0.01, i + bar_height/2, f"{uniform_acc[i]:.2f}", va='center')
    ax.text(entropy_acc[i] + 0.01, i - bar_height/2, f"{entropy_acc[i]:.2f}", va='center')

# Ticks and labels
ax.set_yticks(y)
ax.set_yticklabels(configurations, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('LLM Accuracy', fontsize=12)
ax.set_title('LLM Accuracies: Uniform vs. Entropy Sampling', fontsize=14)
ax.legend(fontsize=10)

plt.tight_layout(pad=4.0)  # Add padding around the plot
plt.savefig('llm_accuracies.png', dpi=600, bbox_inches='tight')
plt.show()
# import pandas as pd
# import matplotlib.pyplot as plt

# # Data aggregation
# configs = [
#     {'Configuration': 'Rank+RAG (R)',          'LLM Acc': 0.3406, 'Reward Acc': 0.3563, 'Invalid Flags': 9},
#     {'Configuration': 'Rank+RAG (NR)',         'LLM Acc': 0.5500, 'Reward Acc': 0.5167, 'Invalid Flags': 21},
#     {'Configuration': 'Rank+RAG NR seg3',      'LLM Acc': 0.4313, 'Reward Acc': 0.2875, 'Invalid Flags': 24},
#     {'Configuration': 'Rank+RAG R seg3',       'LLM Acc': 0.3188, 'Reward Acc': 0.3644, 'Invalid Flags': 6},
#     {'Configuration': 'Rank+RAG R seg10',      'LLM Acc': 0.4099, 'Reward Acc': 0.5248, 'Invalid Flags': 6},
#     {'Configuration': 'Rank+RAG NR seg10',     'LLM Acc': 0.4231, 'Reward Acc': 0.5000, 'Invalid Flags': 24},
#     {'Configuration': 'Rank+NoRAG NR seg5',    'LLM Acc': 0.3833, 'Reward Acc': 0.4083, 'Invalid Flags': 15},
#     {'Configuration': 'Rank+NoRAG R seg5',     'LLM Acc': 0.3591, 'Reward Acc': 0.3364, 'Invalid Flags': 8},
#     {'Configuration': 'NoRank+RAG NR',         'LLM Acc': 0.4375, 'Reward Acc': 0.4062, 'Invalid Flags': 18},
#     {'Configuration': 'NoRank+RAG R',       'LLM Acc': 0.3333, 'Reward Acc': 0.3333, 'Invalid Flags': 6},
# ]

# df = pd.DataFrame(configs)

# # Plotting accuracies
# plt.figure(figsize=(10, 6))
# plt.bar(df['Configuration'], df['LLM Acc'], label='LLM Acc', alpha=0.7)
# plt.bar(df['Configuration'], df['Reward Acc'], label='Reward Acc', alpha=0.7, bottom=df['LLM Acc'])
# plt.xticks(rotation=45, ha='right')
# plt.ylabel('Accuracy')
# plt.title('Balanced Accuracies Across Configurations')
# plt.legend()
# plt.tight_layout()
# plt.savefig('all_accuracies.png')
# plt.close()

# # Plotting invalid flags
# plt.figure(figsize=(10, 6))
# plt.bar(df['Configuration'], df['Invalid Flags'], color='orange')
# plt.xticks(rotation=45, ha='right')
# plt.ylabel('Invalid Flags Count')
# plt.title('Invalid Indices Count Across Configurations')
# plt.tight_layout()
# plt.savefig('all_invalid_flags.png')
# plt.close()





import matplotlib.pyplot as plt
import pandas as pd

# Data preparation
data = {
    "Configuration": [
        "Rank + RAG R", "Rank + RAG NR", "Rank + RAG R, seg=3", "Rank + RAG NR, seg=3",
        "Rank + No RAG R", "Rank + No RAG NR", "No Rank + RAG R", "No Rank + RAG NR",
        "No Rank + No RAG R", "No Rank + No RAG NR"
    ],
    "LLM Label Accuracy": [0.6000, 0.7500, 0.8333, 0.6667, 0.8333, 0.7143, 0.8000, 0.6000, 0.6667, 0.2000],
    "Invalid Responses": [5, 6, 4, 7, 4, 3, 5, 5, 4, 8]
}

df = pd.DataFrame(data)

# Plot 1: Label Accuracy vs Configuration (sorted descending)
df_sorted_acc = df.sort_values(by="LLM Label Accuracy", ascending=False)
plt.figure(figsize=(12, 6))
plt.bar(df_sorted_acc["Configuration"], df_sorted_acc["LLM Label Accuracy"])
plt.xlabel("Configuration")
plt.ylabel("LLM Label Accuracy")
plt.title("LLM Label Accuracy by Configuration (Sorted)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot 2: Invalid Responses vs Configuration (sorted descending)
df_sorted_inv = df.sort_values(by="Invalid Responses", ascending=False)
plt.figure(figsize=(12, 6))
plt.bar(df_sorted_inv["Configuration"], df_sorted_inv["Invalid Responses"])
plt.xlabel("Configuration")
plt.ylabel("Invalid Responses")
plt.title("Invalid Responses by Configuration (Sorted)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


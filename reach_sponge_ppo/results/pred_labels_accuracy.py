import json

# Define the checkpoints corresponding to each line in the JSONL
checkpoints = [1000, 2000, 3500, 4500, 5500, 6500, 7500, 8500, 9500, 10000]

# Prepare output arrays
accuracy            = []
Invalids            = []
TP                  = []
TN                  = []
FP                  = []
FN                  = []
total_true_rewards  = []
total_pred_rewards  = []

# Load each JSON object (one per line) from the .jsonl file
with open("records.jsonl", "r", encoding="utf-8") as fin:
    for line in fin:
        rec = json.loads(line)

        # --- Labels (unchanged) ---
        true_labels = [x[0] for x in rec["true_labels"]]
        pred_labels = [x[0] for x in rec["pred_labels"]]

        # --- Rewards: extract directly, so we catch missing‐key errors ---
        # true_rew_list = rec["true_rew"]
        # pred_rew_list = rec["pred_rew"]
        # true_rew_sum  = sum(true_rew_list)
        # pred_rew_sum  = sum(pred_rew_list)
        # total_true_rewards.append(true_rew_sum)
        # total_pred_rewards.append(pred_rew_sum)

        # --- Classification metrics ---
        total = len(true_labels)
        # Invalids.append(total)

        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)

        TP.append(tp)
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)

        # Compute balanced accuracy: ½ (TPR + TNR)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_acc = 0.5 * (tpr + tnr)
        accuracy.append(balanced_acc)
        

# Print results:
for ck, acc, tp, tn, fp, fn in zip(
        checkpoints,
        accuracy,
        TP, TN, FP, FN,
        # total_true_rewards,
        # total_pred_rewards
    ):
    print(
        f"Step {ck:5d}:  "
        f"TP={tp:3d}, TN={tn:3d}, FP={fp:3d}, FN={fn:3d}, "
        f"BalancedAcc={acc:.4f}, "
        # f"TotalTrueRew={tr:.3f}, TotalPredRew={pr:.3f}"
    )



# import matplotlib.pyplot as plt

# # Data
# checkpoints = [10000, 50000, 100000]
# balanced_acc = [0.5195, 0.5383, 0.5446]


# # Create figure
# fig, ax = plt.subplots(figsize=(8, 5))

# # Bar chart
# ax.bar(checkpoints, balanced_acc, width=8000, alpha=0.6, label="Balanced Accuracy")

# # Line overlay
# ax.plot(checkpoints, balanced_acc, marker='o', linewidth=2, label="Balanced Accuracy")

# # Labels and title
# ax.set_xlabel("Checkpoint")
# ax.set_ylabel("Balanced Accuracy")
# ax.set_title("Balanced Accuracy vs. Checkpoint")
# ax.set_xticks(checkpoints)
# ax.set_xticklabels(["10k", "50k", "100k"])
# ax.set_ylim(0, 1.0)
# ax.grid(axis='y', linestyle='--', alpha=0.5)
# ax.legend()

# # Save
# fig.tight_layout()
# fig.savefig("pref_check.png", dpi=300)

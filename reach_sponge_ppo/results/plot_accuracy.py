import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────
# Given data
# ────────────────────────────────────────────────────────────────────
labels = [
    "no rag\n03mini", "rag\n03mini", "rag\n40mini", "no rag\n40mini",
    "rag\n03mini\nH=3", "no rag\n03mini\nH=3", "rag\n03mini\nH=6", "no rag\n03mini\nH=6",
     "rag\nno rank\n03mini", "no rag\nno rank\n03mini", "rag\nno rank\n04mini"
]

acc_vals_1 = [0.8635, 0.8011, 0.8295, 0.7991, 0.8291, 0.8386, 0.7464, 0.7557, 0.6932, 0.7404, 0.6053]
acc_vals_2 = [0.7381, 0.7443, 0.6964, 0.6528, 0.7229, 0.7400, 0.6000, 0.6250, 0.7473, 0.6798, 0.6857]

invalids1  = [3, 2, 13, 13, 6, 4, 5, 2, 4, 2, 7]
invalids2  = [4, 6, 17, 17, 4, 3, 3, 5, 6, 5, 6]

TP1 = [17, 15, 12, 13, 12, 16, 12, 14, 14, 15, 11,]
TN1 = [21, 21, 10, 11, 20, 19, 19, 21,  15, 19, 12,]
FP1 = [1, 3, 1, 3, 1, 1, 4, 3, 5, 7, 7]
FN1 = [5, 6, 4, 3, 5, 6, 6, 8, 8, 5, 8]

TP2 = [19, 19, 9, 9, 14, 17, 9, 10, 20, 17, 20]
TN2 = [12, 10, 6, 5, 17, 16, 18, 15, 10, 11, 12]
FP2 = [9, 6, 2, 4, 4, 4, 6, 5, 6, 9, 5]
FN2 = [2, 3, 5, 3, 8, 8, 11, 10, 3, 4, 5]

# ────────────────────────────────────────────────────────────────────
# Build DataFrame & compute stats
# ────────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    'label':           labels,
    'accuracy_run1':   acc_vals_1,
    'accuracy_run2':   acc_vals_2,
    'invalids_run1':   invalids1,
    'invalids_run2':   invalids2,
    'TP_run1':         TP1,
    'TN_run1':         TN1,
    'FP_run1':         FP1,
    'FN_run1':         FN1,
    'TP_run2':         TP2,
    'TN_run2':         TN2,
    'FP_run2':         FP2,
    'FN_run2':         FN2,
})

# Mean & std for accuracy
df['accuracy_mean'] = df[['accuracy_run1','accuracy_run2']].mean(axis=1)
df['accuracy_std']  = df[['accuracy_run1','accuracy_run2']].std(axis=1)

# Mean & std for invalids
df['invalids_mean'] = df[['invalids_run1','invalids_run2']].mean(axis=1)
df['invalids_std']  = df[['invalids_run1','invalids_run2']].std(axis=1)

# ────────────────────────────────────────────────────────────────────
# Print the table to console
# ────────────────────────────────────────────────────────────────────
print(df.to_string(index=False))

# ────────────────────────────────────────────────────────────────────
# Plot: Balanced Accuracy
# ────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.bar(df['label'], df['accuracy_mean'], 
        yerr=df['accuracy_std'], capsize=5)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy per Model with Error Bars')
plt.tight_layout()
plt.savefig('./accuracy_plot.png', dpi=600)
plt.close()

# ────────────────────────────────────────────────────────────────────
# Plot: Invalid Indices
# ────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.bar(df['label'], df['invalids_mean'], 
        yerr=df['invalids_std'], capsize=5)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Invalid Indices Count')
plt.title('Invalid Indices per Model with Error Bars')
plt.tight_layout()
plt.savefig('./invalids_plot.png', dpi=600)
plt.close()


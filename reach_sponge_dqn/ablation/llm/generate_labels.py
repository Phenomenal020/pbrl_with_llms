#!/usr/bin/env python3
import json
import sys
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

def extract_and_score(input_path: str, labels_output: str, accs_output: str):

    llm_labels = []
    pred_labels = []
    true_labels = []

    # Extract and save labels
    with open(input_path, 'r') as fin, open(labels_output, 'w') as fout:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {lineno}: JSON decode error: {e}", file=sys.stderr)
                continue

            # Ensure required keys exist
            try:
                llm = obj["llm_label"]
                pred = obj["prediction_label"]
                true = obj["true_label"]
            except KeyError as e:
                print(f"Skipping line {lineno}: missing key {e}", file=sys.stderr)
                continue

            llm_labels.append(llm)
            pred_labels.append(pred)
            true_labels.append(true)

            out = {
                "index":            obj.get("index"),
                "llm_label":        llm,
                "prediction_label": pred,
                "true_label":       true,
            }
            fout.write(json.dumps(out) + "\n")

    # Compute balanced accuracies
    bal_acc_llm  = balanced_accuracy_score(true_labels, llm_labels)
    bal_acc_pred = balanced_accuracy_score(true_labels, pred_labels)

    # Compute confusion matrices
    cm_llm    = confusion_matrix(true_labels, llm_labels)
    cm_pred   = confusion_matrix(true_labels, pred_labels)

    # Assuming binary labels [0,1]: cm = [[TN, FP], [FN, TP]]
    LLM_TN, LLM_FP, LLM_FN, LLM_TP = cm_llm.ravel()
    REWARD_TN, REWARD_FP, REWARD_FN, REWARD_TP = cm_pred.ravel()

    # Write to Python file
    with open(accs_output, 'a') as fa:
        fa.write("# Auto-generated balanced accuracy, confusion matrix, and invalids results\n\n")

        # Balanced accuracies
        fa.write(f"# Balanced accuracies\n")
        fa.write(f"LLM_BALANCED_ACCURACY = {bal_acc_llm:.4f}\n")
        fa.write(f"REWARD_MODEL_BALANCED_ACCURACY = {bal_acc_pred:.4f}\n\n")

        # Confusion matrix counts
        fa.write(f"# LLM confusion matrix counts\n")
        fa.write(f"LLM_TP = {LLM_TP}\n")
        fa.write(f"LLM_TN = {LLM_TN}\n")
        fa.write(f"LLM_FP = {LLM_FP}\n")
        fa.write(f"LLM_FN = {LLM_FN}\n\n")

        fa.write(f"# Reward model confusion matrix counts\n")
        fa.write(f"REWARD_TP = {REWARD_TP}\n")
        fa.write(f"REWARD_TN = {REWARD_TN}\n")
        fa.write(f"REWARD_FP = {REWARD_FP}\n")
        fa.write(f"REWARD_FN = {REWARD_FN}\n\n")

        # Placeholder for invalid flags
        fa.write(f"# Invalid sample flags, in same order as labels.jsonl\n")
        fa.write("INVALID_FLAGS = []\n")

    print(f"Labels extracted to: {labels_output}")
    print(f"Accuracies and confusion counts written to: {accs_output}")

def main():
    input_path  = "preferences.jsonl"
    labels_out  = "labels.jsonl"
    accs_out    = "accuracies.py"
    extract_and_score(input_path, labels_out, accs_out)

if __name__ == "__main__":
    main()

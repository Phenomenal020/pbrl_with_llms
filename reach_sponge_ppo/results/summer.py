import json 
import jsonlines

def main(file):
    # ----- Option B: Read from file (uncomment to use) -----
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    total_collisions = sum(ep["n_collisions"] for ep in data)
    total_true_reward = sum(ep["true reward"] for ep in data)

    print(f"Total collisions: {total_collisions}")
    print(f"Sum of true rewards: {total_true_reward:.6f}")

if __name__ == "__main__":
    file = "./enhanced_rl/enhanced_rl_compare5010.jsonl"
    main(file)
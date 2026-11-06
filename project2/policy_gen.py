import argparse
import csv
import numpy as np

from collections import defaultdict
from pathlib import Path

DATASET = None

CONFIG = {
    "small": {"num_states": 100, "num_actions": 4, "gamma": 0.95},
    "medium": {"num_states": 50000, "num_actions": 7, "gamma": 1.0},
    "large": {"num_states": 302020, "num_actions": 9, "gamma": 0.95},
}

def read_transitions(csv_path):
    transitions = defaultdict(list)

    with open(csv_path) as file:
        reader = csv.DictReader(file)

        for row in reader:
            s = int(row["s"])
            a = int(row["a"])
            r = int(row["r"])
            sp = int(row["sp"])

            transitions[(s, a)].append((r, sp))

    return dict(transitions)

def generate_policy_file(Q, filename):
    with open(filename, "w") as file:
        for action in Q.argmax(axis=1) + 1:
            file.write(f"{action}\n")

def q_iteration(transitions, num_states, num_actions, gamma=0.95, max_iterations=10000,
                tolerance=1e-6):
    Q_prev = np.zeros((num_states, num_actions))

    print(f"Starting Q-iteration: gamma={gamma}, tolerance={tolerance}, "
          f"max_iterations={max_iterations}")

    for iteration in range(1, max_iterations + 1):
        Q_new = Q_prev.copy()
        next_values = Q_prev.max(axis=1)

        for (s, a), samples in transitions.items():
            summation = 0

            for sample in samples:
                r, sp = sample

                if DATASET == "medium" and r > 0:
                    summation += r
                else:
                    summation += r + gamma * next_values[sp - 1]

            Q_new[s - 1, a - 1] = summation / len(samples)

        change = np.max(np.abs(Q_new - Q_prev))
        Q_prev = Q_new
        print(f"Iteration {iteration}: max Q change = {change:.6g}")

        if change < tolerance:
            print(f"Converged after {iteration} iterations.")
            break
    else:
        print("Reached the iteration limit without convergence.")

    return Q_prev

def main():
    global DATASET

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=CONFIG)
    args = parser.parse_args()
    DATASET = args.dataset

    project_dir = Path(__file__).resolve().parent
    csv_path = project_dir / "data" / f"{DATASET}.csv"
    policy_path = project_dir / f"{DATASET}.policy"

    transitions = read_transitions(csv_path)

    Q = q_iteration(transitions, **CONFIG[DATASET])

    generate_policy_file(Q, policy_path)

if __name__ == "__main__":
    main()

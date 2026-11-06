import argparse
import csv
import numpy as np

from collections import defaultdict
from pathlib import Path

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

def q_iteration(transitions, num_states, num_actions, gamma=0.95, iterations=10):
    Q_prev = np.zeros((num_states, num_actions))

    for _ in range(iterations):
        Q_new = Q_prev.copy()
        next_values = Q_prev.max(axis=1)

        for (s, a), samples in transitions.items():
            summation = 0

            for sample in samples:
                r, sp = sample
                summation += r + gamma * next_values[sp - 1]

            Q_new[s - 1, a - 1] = summation / len(samples)

        Q_prev = Q_new

    return Q_prev

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=CONFIG)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    csv_path = project_dir / "data" / f"{args.dataset}.csv"
    policy_path = project_dir / f"{args.dataset}.policy"

    transitions = read_transitions(csv_path)

    Q = q_iteration(transitions, **CONFIG[args.dataset])

    generate_policy_file(Q, policy_path)

if __name__ == "__main__":
    main()

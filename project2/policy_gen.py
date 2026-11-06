import argparse
import csv

from collections import defaultdict
from pathlib import Path

def read_transitions(csv_path):
    transitions = defaultdict(list)
    states = set()
    actions = set()

    with open(csv_path) as file:
        reader = csv.DictReader(file)

        for row in reader:
            s = int(row["s"])
            a = int(row["a"])
            r = int(row["r"])
            sp = int(row["sp"])

            transitions[(s, a)].append((r, sp))
            states.update([s, sp])
            actions.add(a)

    return dict(transitions), states, actions

def generate_policy_file(Q, filename, states):
    with open(filename, "w") as file:
        for state in states:
            if state not in Q:
                raise ValueError("state not in Q")

            action_values = Q[state]
            action = max(sorted(action_values), key=action_values.get)
            file.write(f"{action}\n")

def q_iteration(transitions, states, actions, gamma=0.95, iterations=10):
    Q_prev = {
        state: {
            action: 0.0 for action in actions
        } for state in states
    }

    for _ in range(iterations):
        Q_new = {
            state: values.copy() for state, values in Q_prev.items()
        }

        for (s, a), samples in transitions.items():
            summation = 0

            for sample in samples:
                r, sp = sample
                summation += r + gamma * max(Q_prev[sp].values())

            Q_new[s][a] = summation / len(samples)

        Q_prev = Q_new

    return Q_prev

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["small", "medium", "large"])
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    csv_path = project_dir / "data" / f"{args.dataset}.csv"
    policy_path = project_dir / f"{args.dataset}.policy"

    transitions, states, actions = read_transitions(csv_path)

    Q = q_iteration(transitions, states, actions)

    generate_policy_file(Q, policy_path, len(states))


if __name__ == "__main__":
    main()

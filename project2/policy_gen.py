import argparse
import csv
import numpy as np

from collections import defaultdict

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

    counts = {
        pair: len(sample) for pair, sample in transitions.items()
    }

    return dict(transitions), states, actions, counts

def q_iteration(transitions, states, actions, counts, gamma=0.95, iterations=10):
    Q_prev = np.zeros((len(states), len(actions)))

    for i in range(iterations):
        Q_new = Q_prev.copy()

        for (s, a), samples in transitions.items():
            summation = 0

            for sample in samples:
                r, sp = sample
                summation += r + gamma * Q_prev[sp - 1].max()

            Q_new[s - 1, a - 1] = summation / len(samples)

        Q_prev = Q_new

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()

    transitions, states, actions, counts = read_transitions(args.csv_path)

    q_iteration(transitions, states, actions, counts)


if __name__ == "__main__":
    main()

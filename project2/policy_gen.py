import argparse
import csv
import numpy as np

from collections import defaultdict
from pathlib import Path

DATASET = None

# For medium.csv
POSITION_BIN_WIDTH = 5
VELOCITY_BIN_WIDTH = 2

CONFIG = {
    "small": {"num_states": 100, "num_actions": 4, "gamma": 0.95},
    "medium": {"num_states": 50000, "num_actions": 7, "gamma": 1.0},
    "large": {"num_states": 302020, "num_actions": 9, "gamma": 0.95},
}

def coarse_state(state):
    position = (state - 1) % 500
    velocity = (state - 1) // 500
    num_positions = (500 + POSITION_BIN_WIDTH - 1) // POSITION_BIN_WIDTH
    return 1 + position // POSITION_BIN_WIDTH + num_positions * (velocity // VELOCITY_BIN_WIDTH)

def read_transitions(csv_path):
    transitions = defaultdict(list)

    with open(csv_path) as file:
        reader = csv.DictReader(file)

        for row in reader:
            s = int(row["s"])
            a = int(row["a"])
            r = int(row["r"])
            sp = int(row["sp"])

            if DATASET == "medium":
                s = coarse_state(s)
                sp = coarse_state(sp)

            transitions[(s, a)].append((r, sp))

    return dict(transitions)

def generate_policy_file(Q, filename):
    policy = Q.argmax(axis=1) + 1
    with open(filename, "w") as file:
        for state in range(1, CONFIG[DATASET]["num_states"] + 1):
            row = coarse_state(state) - 1 if DATASET == "medium" else state - 1
            file.write(f"{policy[row]}\n")

def missing_pair_neighbors(transitions, num_states, num_actions):
    # Mark pairs that have actual transition samples.
    observed = np.zeros((num_states, num_actions), dtype=bool)

    for s, a in transitions:
        observed[s - 1, a - 1] = True

    num_positions = (500 + POSITION_BIN_WIDTH - 1) // POSITION_BIN_WIDTH
    num_velocities = (100 + VELOCITY_BIN_WIDTH - 1) // VELOCITY_BIN_WIDTH
    rows = np.arange(num_states)
    positions = (rows % num_positions) / num_positions
    velocities = (rows // num_positions) / num_velocities

    neighbors = []

    for a in range(num_actions):
        # Ue states with samples for this action as neighbors
        sources = np.flatnonzero(observed[:, a])

        if len(sources) == 0:
            raise ValueError(f"No observed samples for action {a + 1}")

        for row in np.flatnonzero(~observed[:, a]):
            distances = ((positions[sources] - positions[row]) ** 2 + (velocities[sources] - velocities[row]) ** 2)
            nearest = sources[distances.argmin()]
            neighbors.append((row, a, nearest))

    return neighbors

def q_iteration(transitions, num_states, num_actions, gamma=0.95, max_iterations=5000,
                tolerance=1e-6):
    Q_prev = np.zeros((num_states, num_actions))
    neighbors = (missing_pair_neighbors(transitions, num_states, num_actions)
                 if DATASET == "medium" else [])

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

        for row, a, source in neighbors:
            Q_new[row, a] = Q_new[source, a]

        change = np.max(np.abs(Q_new - Q_prev))
        policy_changes = np.count_nonzero(Q_new.argmax(axis=1) != Q_prev.argmax(axis=1))
        Q_prev = Q_new
        print(f"Iteration {iteration}: max Q change = {change:.6g}, "
              f"policy changes = {policy_changes}")

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

    training_config = CONFIG[DATASET].copy()

    if DATASET == "medium":
        training_config["num_states"] = coarse_state(CONFIG[DATASET]["num_states"])

    Q = q_iteration(transitions, **training_config)

    generate_policy_file(Q, policy_path)

if __name__ == "__main__":
    main()

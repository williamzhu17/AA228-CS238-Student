import argparse
import csv
import numpy as np

from collections import defaultdict
from pathlib import Path
from time import perf_counter

DATASET = None

# For medium.csv
POSITION_BIN_WIDTH = 5
VELOCITY_BIN_WIDTH = 2
MEDIUM_GOAL_POSITION = 466
MEDIUM_ACTION_COSTS = [-225, -100, -25, 0, -25, -100, -225]

CONFIG = {
    "small": {"num_states": 100, "num_actions": 4, "gamma": 0.95},
    "medium": {"num_states": 50000, "num_actions": 7, "gamma": 1.0},
    "large": {"num_states": 302020, "num_actions": 9, "gamma": 0.95},
}

def decode_state(state):
    """Return medium's position and velocity bin indices (not physical units)."""
    position = (state - 1) % 500
    velocity = (state - 1) // 500
    return position, velocity

def coarse_state(state):
    position, velocity = decode_state(state)
    num_positions = (500 + POSITION_BIN_WIDTH - 1) // POSITION_BIN_WIDTH
    return 1 + position // POSITION_BIN_WIDTH + num_positions * (velocity // VELOCITY_BIN_WIDTH)

def read_transitions(csv_path):
    transitions = defaultdict(list)
    movements = defaultdict(list)

    with open(csv_path) as file:
        reader = csv.DictReader(file)

        for row in reader:
            s = int(row["s"])
            a = int(row["a"])
            r = int(row["r"])
            sp = int(row["sp"])

            if DATASET == "medium":
                position, velocity = decode_state(s)
                next_position, next_velocity = decode_state(sp)

                delta_position = next_position - position
                delta_velocity = next_velocity - velocity

                s = coarse_state(s)
                sp = coarse_state(sp)
                movements[(s, a)].append((delta_position, delta_velocity))

            transitions[(s, a)].append((r, sp))

    transitions = dict(transitions)

    # Since medium is sparse, preprocess generates synthetic observations based on nearby observations
    if DATASET == "medium":
        transitions = preprocess_medium(transitions, movements)

    return transitions

def generate_policy_file(Q, filename):
    policy = Q.argmax(axis=1) + 1
    with open(filename, "w") as file:
        for state in range(1, CONFIG[DATASET]["num_states"] + 1):
            row = coarse_state(state) - 1 if DATASET == "medium" else state - 1
            file.write(f"{policy[row]}\n")

def preprocess_medium(transitions, movements):
    num_states = coarse_state(CONFIG["medium"]["num_states"])
    num_actions = CONFIG["medium"]["num_actions"]
    observed = np.zeros((num_states, num_actions), dtype=bool)

    for s, a in transitions:
        observed[s - 1, a - 1] = True

    num_positions = (500 + POSITION_BIN_WIDTH - 1) // POSITION_BIN_WIDTH
    num_velocities = (100 + VELOCITY_BIN_WIDTH - 1) // VELOCITY_BIN_WIDTH
    rows = np.arange(num_states)
    positions = (rows % num_positions) / num_positions
    velocities = (rows // num_positions) / num_velocities

    filled = transitions.copy()

    for a in range(num_actions):
        # Use states with samples for this action as neighbors
        sources = np.flatnonzero(observed[:, a])

        if len(sources) == 0:
            raise ValueError(f"No observed samples for action {a + 1}")

        for row in np.flatnonzero(~observed[:, a]):
            distances = ((positions[sources] - positions[row]) ** 2 + (velocities[sources] - velocities[row]) ** 2)
            k = min(3, len(sources))
            nearest = sources[np.argpartition(distances, k - 1)[:k]]

            # Use the center of bin
            p_min = (row % num_positions) * POSITION_BIN_WIDTH
            v_min = (row // num_positions) * VELOCITY_BIN_WIDTH
            position = (p_min + min(p_min + POSITION_BIN_WIDTH - 1, 499)) / 2
            velocity = (v_min + min(v_min + VELOCITY_BIN_WIDTH - 1, 99)) / 2

            samples = []

            for source in nearest:
                for dp, dv in movements[(source + 1, a + 1)]:
                    # Need to clip of out of bounds
                    next_position = np.clip(position + dp, 0, 499)
                    next_velocity = np.clip(velocity + dv, 0, 99)

                    terminal = next_position >= MEDIUM_GOAL_POSITION
                    reward = MEDIUM_ACTION_COSTS[a] + (100000 if terminal else 0)

                    next_state = 1 + int(next_position) + 500 * int(next_velocity)
                    samples.append((reward, coarse_state(next_state)))
            filled[(row + 1, a + 1)] = samples

    print(f"Preprocessed medium: filled {len(filled) - len(transitions)} missing pairs.")
    return filled

def q_iteration(transitions, num_states, num_actions, gamma=0.95, max_iterations=1000,
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
    start_time = perf_counter()
    main()
    print(f"Total runtime: {perf_counter() - start_time:.2f} seconds")

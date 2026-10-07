import networkx
import numpy as np
import os
import pandas as pd
import sys
import time

from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from scipy.special import gammaln

def read_gph(names2idx, filename):
    G = networkx.DiGraph()
    G.add_nodes_from(range(len(names2idx)))

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parent, child = [s.strip() for s in line.split(",")]
            G.add_edge(names2idx[parent], names2idx[child])

    return G

def write_gph(dag, idx2names, filename):
    edges = sorted((idx2names[u], idx2names[v]) for u, v in dag.edges())
    with open(filename, 'w') as f:
        for parent, child in edges:
            f.write("{}, {}\n".format(parent, child))

def process_csv(path):
    """Process CSV to return data, cardinalities, idx2name dict, names2idx dict"""
    df = pd.read_csv(path)
    names = list(df.columns)

    idx2names = {}
    names2idx = {}

    for i, name in enumerate(names):
        idx2names[i] = name
        names2idx[name] = i

    data = df.to_numpy(dtype=int) - 1
    r = data.max(axis=0) + 1

    return data, r, idx2names, names2idx

###############################################################################
# Compute Bayesian Score
###############################################################################

def parent_config_index(row, parents, r):
    """Map one sample's parent values to single j"""
    if len(parents) == 0:
        return 0

    parent_values = tuple(row[p] for p in parents)
    parent_dims = tuple(r[p] for p in parents)
    return int(np.ravel_multi_index(parent_values, parent_dims))

def count_ijk(i, parents, data, r, j=None):
    """Build count table M for variable X_i"""
    parents = list(parents)
    r_i = int(r[i])
    n = data.shape[0]

    if not parents:
        q_i = 1
        if j is None:
            j = np.zeros(n, dtype=int)
    else:
        parent_dims = tuple(int(r[p]) for p in parents)
        q_i = int(np.prod(parent_dims))
        if j is None:
            j = np.ravel_multi_index(data[:, parents].T, parent_dims)

    M = np.bincount(j * r_i + data[:, i], minlength=q_i * r_i).reshape(q_i, r_i)

    return M

_score_cache = {}

def local_score(i, parents, data, r, j=None):
    """Bayesian score for X_i given parents"""
    parents = list(parents)

    # Check if in cache
    key = (i, frozenset(parents))

    if key in _score_cache:
        return _score_cache[key]
    
    # Compute
    M = count_ijk(i, parents, data, r, j=j)
    r_i = r[i]

    a_ij0 = r_i
    m_ij0 = M.sum(axis=1)

    observed = m_ij0 > 0
    M = M[observed]
    m_ij0 = m_ij0[observed]

    score = np.sum(
        gammaln(a_ij0) - gammaln(a_ij0 + m_ij0) + np.sum(gammaln(1 + M), axis=1)
    )

    # Add to cache
    _score_cache[key] = score

    return float(score)

def bayesian_score(graph, data, r):
    i_scores = [local_score(i, list(graph.predecessors(i)), data, r) for i in graph.nodes()]

    return sum(i_scores)

def update_local_score(graph, i, data, r, local_scores, j=None):
    new = local_score(i, graph.predecessors(i), data, r, j=j)
    delta = new - local_scores[i]
    local_scores[i] = new

    return delta

def randomize_graph(graph, p=0.1, max_parents=10):
    """Initialize random DAG"""
    _score_cache.clear()
    graph.remove_edges_from(list(graph.edges()))

    n = graph.number_of_nodes()
    order = np.random.permutation(n)

    for a in range(n):
        for b in range(a + 1, n):
            child = int(order[b])

            if graph.in_degree(child) >= max_parents:
                continue

            if np.random.random() < p:
                graph.add_edge(int(order[a]), child)

    return graph

def explore_loop(graph, data, r, local_scores, trials=1000, tabu_tenure=10, patience=50, max_parents=10, improve_eps=1e-4):
    """
    Explore graphs
    Explore all possible valid moves and choose the one with the best delta
    Have tabu list to escape local max
    Have aspiration to allow a tabu move if it beats global max
    Restart after patience with no improvement
    """
    best_score = sum(local_scores)
    global_best_score = best_score
    global_best_edges = list(graph.edges())
    nodes = list(graph.nodes())
    n = len(nodes)
    t0 = time.perf_counter()

    # Initialize j_cache
    j_cache = []
    for i in range(n):
        parents = list(graph.predecessors(i))
        if parents:
            j_cache.append(
                np.ravel_multi_index(
                    data[:, parents].T, tuple(int(r[p]) for p in parents)
                )
            )
        else:
            j_cache.append(np.zeros(data.shape[0], dtype=int))

    tabu = deque(maxlen=tabu_tenure)
    stale = 0

    trial = 0
    while trial < trials:
        edges = list(graph.edges())
        candidates = []

        # add edge candidates - 0 = add
        for u in range(n):
            for v in range(n):
                # Skip same nodes, existing edges, parent cap, or edges that create cycles
                if (u == v or graph.has_edge(u, v) or graph.in_degree(v) >= max_parents or networkx.has_path(graph, v, u)):
                    continue

                candidates.append((0, u, v))

        # delete edge candidates - 1 = delete
        for u, v in edges:
            candidates.append((1, u, v))

        # flip edge candidates - 2 = flip
        for u, v in edges:
            # Flip gives u a new parent
            if graph.in_degree(u) >= max_parents:
                continue

            graph.remove_edge(u, v)

            if not networkx.has_path(graph, u, v):
                candidates.append((2, u, v))

            graph.add_edge(u, v)

        best_move = None
        best_delta = -np.inf
        best_j_new = None
        best_local_scores = None

        for move, u, v in candidates:
            if move == 0:       # add edge
                graph.add_edge(u, v)
                affected = [v]
            elif move == 1:     # delete edge
                graph.remove_edge(u, v)
                affected = [v]
            else:               # flip edge
                graph.remove_edge(u, v)
                graph.add_edge(v, u)
                affected = [u, v]

            # Update and see if score improved
            prev_local_scores = {i: local_scores[i] for i in affected}
            prev_j = {i: j_cache[i] for i in affected}
            j_new = {}
            delta = 0.0

            for i in affected:
                if move == 0:
                    # add increments j
                    j_new[i] = j_cache[i] * int(r[u]) + data[:, u]
                    delta += update_local_score(graph, i, data, r, local_scores, j=j_new[i])
                elif move == 2 and i == u:
                    # on flip, u gains parent and can increment j cache
                    j_new[i] = j_cache[i] * int(r[v]) + data[:, v]
                    delta += update_local_score(graph, i, data, r, local_scores, j=j_new[i])
                else: 
                    delta += update_local_score(graph, i, data, r, local_scores)

                    # Reset j cache
                    parents = list(graph.predecessors(i))
                    if parents:
                        j_new[i] = np.ravel_multi_index(
                            data[:, parents].T, tuple(int(r[p]) for p in parents)
                        )
                    else:
                        j_new[i] = np.zeros(data.shape[0], dtype=int)

            # Skip tabu unless aspiration
            allowed = (move, u, v) not in tabu or best_score + delta > global_best_score

            # Track best before undoing
            if allowed and delta > best_delta:
                best_delta = delta
                best_move = (move, u, v)
                best_j_new = j_new
                best_local_scores = {i: local_scores[i] for i in affected}

            # Undo current move
            if move == 0:
                graph.remove_edge(u, v)
            elif move == 1:
                graph.add_edge(u, v)
            else:
                graph.remove_edge(v, u)
                graph.add_edge(u, v)

            # Undo caches
            for i in affected:
                local_scores[i] = prev_local_scores[i]
                j_cache[i] = prev_j[i]

        if best_move is None:
            break

        # Save best move
        move, u, v = best_move

        if move == 0:
            graph.add_edge(u, v)
            tabu.append((1, u, v))
        elif move == 1:
            graph.remove_edge(u, v)
            tabu.append((0, u, v))
        else:
            graph.remove_edge(u, v)
            graph.add_edge(v, u)
            tabu.append((2, v, u))

        for i, s in best_local_scores.items():
            local_scores[i] = s
        for i, j in best_j_new.items():
            j_cache[i] = j

        best_score += best_delta

        # Require significant improvement so patience can trigger restarts
        if best_score > global_best_score + improve_eps:
            global_best_score = best_score
            global_best_edges = list(graph.edges())
            stale = 0
        else:
            stale += 1

            if stale >= patience:
                break

        trial += 1

    graph.remove_edges_from(list(graph.edges()))
    graph.add_edges_from(global_best_edges)

    elapsed = time.perf_counter() - t0
    return graph, global_best_score, trial, elapsed

def run_random_restart(data, r, seed, trials=10000, max_parents=10):
    """Worker function for random restart hill climb"""
    np.random.seed(seed)
    _score_cache.clear()

    G = networkx.DiGraph()
    G.add_nodes_from(range(data.shape[1]))
    randomize_graph(G)

    local_scores = [local_score(i, list(G.predecessors(i)), data, r) for i in G.nodes()]
    G, score, used, elapsed = explore_loop(G, data, r, local_scores, trials=trials, max_parents=max_parents)

    return score, list(G.edges()), used, elapsed

def explore(infile, outfile, trials=10000, n_workers=None):
    data, r, idx2names, names2idx = process_csv(infile)

    if n_workers is None:
        n_workers = os.cpu_count() - 1 or 4

    # Initialize empty graph
    G = networkx.DiGraph()
    G.add_nodes_from(range(data.shape[1]))

    global_best_score = -np.inf
    global_best_edges = []
    trials_used = 0
    restart = 0
    max_parents = 15

    # First run with empty graph and single process
    _score_cache.clear()
    G.remove_edges_from(list(G.edges()))

    local_scores = [local_score(i, list(G.predecessors(i)), data, r) for i in G.nodes()]
    G, score, used, elapsed = explore_loop(
        G, data, r, local_scores, trials=trials, max_parents=max_parents
    )

    trials_used += max(used, 1)
    global_best_score = score
    global_best_edges = list(G.edges())
    rate = used / elapsed if elapsed > 0 else 0.0
    remaining = max(0, trials - trials_used)
    restart = 1

    print(
        f"[empty] saturated after {used} steps, "
        f"score={score:.4f}, best={global_best_score:.4f}, "
        f"{elapsed:.2f}s, {rate:.1f} trials/s, remaining={remaining}"
    )

    # Parallel section
    if trials_used <= trials:
        base_seed = int(np.random.randint(0, 2**31 - 1))
        futures = {}

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            def run_one():
                nonlocal restart
                fut = pool.submit(
                    run_random_restart,
                    data,
                    r,
                    base_seed + restart,
                    trials,
                    max_parents,
                )

                futures[fut] = restart
                restart += 1

            while futures or trials_used <= trials:
                while len(futures) < n_workers and trials_used <= trials:
                    run_one()

                if not futures:
                    break

                done, _ = wait(futures, return_when=FIRST_COMPLETED)

                for fut in done:
                    rid = futures.pop(fut)
                    score, edges, used, elapsed = fut.result()
                    trials_used += max(used, 1)

                    if score > global_best_score:
                        global_best_score = score
                        global_best_edges = edges

                    rate = used / elapsed if elapsed > 0 else 0.0
                    remaining = max(0, trials - trials_used)

                    print(
                        f"[restart {rid}] saturated after {used} steps, "
                        f"score={score:.4f}, best={global_best_score:.4f}, "
                        f"{elapsed:.2f}s, {rate:.1f} trials/s, remaining={remaining}"
                    )

    G.remove_edges_from(list(G.edges()))
    G.add_edges_from(global_best_edges)

    write_gph(G, idx2names, outfile)
    print("Score after optimization:", global_best_score)

def main():
    if len(sys.argv) >= 4 and sys.argv[3] == "--score":
        data, r, _, names2idx = process_csv(sys.argv[1])
        G = read_gph(names2idx, sys.argv[2])
        print(bayesian_score(G, data, r))
        return

    if len(sys.argv) not in (3, 4):
        raise Exception(
            "usage: python project1.py <infile>.csv <outfile>.gph [trials]\n"
            "       python project1.py <infile>.csv <graph>.gph --score"
        )

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    trials = int(sys.argv[3]) if len(sys.argv) == 4 else 10000
    explore(inputfilename, outputfilename, trials=trials)


if __name__ == '__main__':
    main()

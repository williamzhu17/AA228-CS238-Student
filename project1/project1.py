import networkx
import numpy as np
import pandas as pd
from scipy.special import gammaln
import sys
import time

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
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

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

def count_ijk(i, parents, data, r):
    """Build count table M for variable X_i"""
    parents = list(parents)
    r_i = int(r[i])
    n = data.shape[0]

    if not parents:
        q_i = 1
        j = np.zeros(n, dtype=int)
    else:
        parent_dims = tuple(int(r[p]) for p in parents)
        q_i = int(np.prod(parent_dims))
        j = np.ravel_multi_index(data[:, parents].T, parent_dims)

    M = np.bincount(j * r_i + data[:, i], minlength=q_i * r_i).reshape(q_i, r_i)

    return M

def local_score(i, parents, data, r):
    """Bayesian score for X_i given parents"""
    parents = list(parents)

    M = count_ijk(i, parents, data, r)
    r_i = r[i]

    a_ij0 = r_i
    m_ij0 = M.sum(axis=1)

    observed = m_ij0 > 0
    M = M[observed]
    m_ij0 = m_ij0[observed]

    score = np.sum(
        gammaln(a_ij0) - gammaln(a_ij0 + m_ij0) + np.sum(gammaln(1 + M), axis=1)
    )

    return float(score)

def bayesian_score(graph, data, r):
    i_scores = [local_score(i, list(graph.predecessors(i)), data, r) for i in graph.nodes()]

    return sum(i_scores)

def update_local_score(graph, i, data, r, local_scores):
    new = local_score(i, graph.predecessors(i), data, r)
    delta = new - local_scores[i]
    local_scores[i] = new

    return delta

def explore_loop(graph, data, r, local_scores, trials=1000):
    """
    Explore graphs
    Add random edge. If it improves score, keep it. If not, throw it out.
    """
    best_score = sum(local_scores)
    nodes = list(graph.nodes())
    n = len(nodes)
    t0 = time.perf_counter()

    for trial in range(trials):
        if (trial + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t0
            rate = (trial + 1) / elapsed
            print(f"Trial {trial + 1}/{trials}, best_score={best_score}, {rate:.1f} trials/s")

        # Pick a random directed edge that is not already present
        u, v = np.random.randint(0, n, size=2)
        if u == v or graph.has_edge(u, v):
            continue

        # Reject if creates a cycle
        # Cycle iff there is already a path from v to u
        if networkx.has_path(graph, v, u):
            continue

        graph.add_edge(u, v)

        # Update and see if score improved
        prev_local_score = local_scores[v]
        delta = update_local_score(graph, v, data, r, local_scores)
        score = best_score + delta

        if score > best_score:
            best_score = score
        else:
            graph.remove_edge(u, v)
            local_scores[v] = prev_local_score

    return graph, best_score

def explore(infile, outfile):
    data, r, idx2names, names2idx = process_csv(infile)

    # Initialize empty graph
    G = networkx.DiGraph()
    G.add_nodes_from(range(data.shape[1]))

    local_scores = [local_score(i, [], data, r) for i in G.nodes()]
    initial_score = sum(local_scores)
    print("Initial score:", initial_score)

    G, score = explore_loop(G, data, r, local_scores, trials=10000)

    write_gph(G, idx2names, outfile)
    print("Score after optimization:", score)

def main():
    if len(sys.argv) == 4 and sys.argv[3] == "--score":
        data, r, _, names2idx = process_csv(sys.argv[1])
        G = read_gph(names2idx, sys.argv[2])
        print(bayesian_score(G, data, r))
        return

    if len(sys.argv) != 3:
        raise Exception(
            "usage: python project1.py <infile>.csv <outfile>.gph\n"
            "       python project1.py <infile>.csv <graph>.gph --score"
        )

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    explore(inputfilename, outputfilename)


if __name__ == '__main__':
    main()

import networkx
import numpy as np
import pandas as pd
from scipy.special import gammaln
import sys

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

###############################################################################
# Compute Bayesian Score
###############################################################################

def parent_config_index(row, parents, r):
    """Map one sample's parent values to single j"""
    if len(parents) == 0:
        return 0

    parent_values = tuple(row[p] - 1 for p in parents)
    parent_dims = tuple(r[p] for p in parents)
    return int(np.ravel_multi_index(parent_values, parent_dims))

def count_ijk(i, parents, data, r):
    """Build count table M for variable X_i"""
    r_i = int(r[i])
    q_i = int(np.prod([r[p] for p in parents])) if parents else 1

    M = np.zeros((q_i, r_i), dtype=int)

    for row in data:
        j = parent_config_index(row, parents, r)
        k = row[i] - 1
        M[j, k] += 1

    return M

def local_score(i, parents, data, r):
    """Bayesian score for X_i given parents"""
    parents = list(parents)

    M = count_ijk(i, parents, data, r)
    r_i = r[i]

    a_ij0 = r_i
    m_ij0 = M.sum(axis=1)

    score = np.sum(
        gammaln(a_ij0) - gammaln(a_ij0 + m_ij0) + np.sum(gammaln(1 + M), axis=1)
    )

    return float(score)

def bayesian_score(graph, data, r):
    i_scores = [local_score(i, list(graph.predecessors(i)), data, r) for i in graph.nodes()]

    return sum(i_scores)

def process_csv(path):
    """Process CSV to return data, cardinalities, idx2name dict, names2idx dict"""
    df = pd.read_csv(path)
    names = list(df.columns)

    idx2names = {}
    names2idx = {}

    for i, name in enumerate(names):
        idx2names[i] = name
        names2idx[name] = i
    
    data = df.to_numpy(dtype=int)
    r = data.max(axis=0)

    return data, r, idx2names, names2idx

def explore_loop(graph, data, r, initial_score, trials=1000):
    """
    Explore graphs
    Add random edge. If it improves score, keep it. If not, throw it out.
    """
    best_score = initial_score
    nodes = list(graph.nodes())
    n = len(nodes)

    for trial in range(trials):
        if (trial + 1) % 1000 == 0:
            print(f"Trial {trial + 1}/{trials}, best_score={best_score}")

        # Pick a random directed edge that is not already present
        u, v = np.random.randint(0, n, size=2)
        if u == v or graph.has_edge(u, v):
            continue

        graph.add_edge(u, v)

        # Reject if it creates a cycle
        if not networkx.is_directed_acyclic_graph(graph):
            graph.remove_edge(u, v)
            continue

        score = bayesian_score(graph, data, r)

        if score > best_score:
            best_score = score
        else:
            graph.remove_edge(u, v)

    return graph, best_score

def explore(infile, outfile):
    data, r, idx2names, names2idx = process_csv(infile)

    # Initialize empty graph
    G = networkx.DiGraph()
    G.add_nodes_from(range(data.shape[1]))

    initial_score = bayesian_score(G, data, r)
    print("Initial score:", initial_score)

    G, score = explore_loop(G, data, r, initial_score, 10000)

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

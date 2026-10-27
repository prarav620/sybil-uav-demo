"""Spatiotemporal graph over (vehicle, window) nodes + node2vec embeddings.

Edges: temporal = a vehicle's consecutive windows; spatial = in the same window, each
node to its k nearest neighbours within `radius` metres (claimed window-mean positions).
No labels are used, so embeddings can be computed over all nodes before the train/test
split without leaking labels.
"""
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def build_graph(fv, radius=150.0, k=5):
    fv = fv.reset_index(drop=True)
    G = nx.Graph()
    G.add_nodes_from(range(len(fv)))
    # temporal edges
    srt = fv.sort_values(["sender", "window"])
    same = (srt["sender"].to_numpy()[1:] == srt["sender"].to_numpy()[:-1]) & \
           (srt["window"].to_numpy()[1:] == srt["window"].to_numpy()[:-1] + 1)
    idx = srt.index.to_numpy()
    G.add_edges_from(zip(idx[:-1][same], idx[1:][same]))
    # spatial edges per window
    for _, grp in fv.groupby("window"):
        if len(grp) < 2:
            continue
        pts = grp[["x_mean", "y_mean"]].to_numpy()
        kk = min(k + 1, len(grp))
        dist, nbr = cKDTree(pts).query(pts, k=kk, distance_upper_bound=radius)
        ids = grp.index.to_numpy()
        for i in range(len(grp)):
            for d, j in zip(dist[i, 1:], nbr[i, 1:]):
                if np.isfinite(d):
                    G.add_edge(ids[i], ids[j])
    return G


def node2vec_embeddings(G, dimensions=32, walk_length=20, num_walks=10, workers=4, seed=0):
    """Return DataFrame indexed by node id with columns emb_0..emb_{d-1}."""
    from node2vec import Node2Vec
    n2v = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks,
                   workers=workers, seed=seed, quiet=True)
    model = n2v.fit(window=5, min_count=1, batch_words=4, seed=seed, workers=workers)
    emb = np.vstack([model.wv[str(n)] for n in G.nodes()])
    return pd.DataFrame(emb, index=list(G.nodes()),
                        columns=[f"emb_{i}" for i in range(dimensions)])

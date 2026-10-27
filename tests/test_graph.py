import numpy as np
import pandas as pd

from src.classifiers.graph_gbdt import feature_columns, train_graph_classifier
from src.graph_builder import build_graph, node2vec_embeddings


def _fv():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(40):
        for w in range(4):
            a = int(s % 3 == 0)
            rows.append(dict(sender=s, window=w, x_mean=10.0 * s, y_mean=0.0, base_trust=0.9 - 0.5 * a + rng.normal(0, .05),
                             sim_claim_fused_dist_med=100.0 * a, is_attacker=a, attack_type=3 * a))
    return pd.DataFrame(rows)


def test_graph_edges_and_embeddings():
    fv = _fv()
    G = build_graph(fv, radius=25.0, k=2)
    assert G.number_of_nodes() == len(fv)
    assert G.has_edge(0, 1)  # same sender, consecutive windows
    e = node2vec_embeddings(G, dimensions=8, walk_length=5, num_walks=2, workers=1)
    assert e.shape == (len(fv), 8)


def test_classifier_runs_and_fusion_flag():
    fv = _fv()
    assert "sim_claim_fused_dist_med" not in feature_columns(fv, use_fusion=False)
    model, test, cols = train_graph_classifier(fv, None, n_estimators=20)
    assert len(test) > 0 and set(cols) >= {"base_trust"}

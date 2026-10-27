"""python src/train.py --variant baseline [--sender-frac 0.2]"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classifiers.baseline_rf import FEATURES, train_baseline  # noqa: E402

DATA = Path("data/processed/merged.parquet")


def load(sender_frac, seed):
    df = pd.read_parquet(DATA)
    if sender_frac < 1.0:  # subsample whole senders so message sequences stay intact
        rng = np.random.default_rng(seed)
        s = df["sender"].unique()
        df = df[df["sender"].isin(rng.choice(s, int(len(s) * sender_frac), replace=False))]
    return df


def train_graph(a):
    from sklearn.metrics import f1_score
    from src.classifiers.graph_gbdt import train_graph_classifier
    from src.graph_builder import build_graph, node2vec_embeddings
    fv = pd.read_parquet("data/processed/features.parquet")  # run `python -m src.features` first
    emb = None
    if not a.no_graph or a.compare:
        G = build_graph(fv)
        print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        emb = node2vec_embeddings(G, seed=a.seed)
    combos = ([(f, g) for f in (False, True) for g in (False, True)] if a.compare
              else [(not a.no_fusion, not a.no_graph)])
    rows = []
    for fusion, graph in combos:
        model, test, cols = train_graph_classifier(fv, emb if graph else None, use_fusion=fusion, seed=a.seed)
        proba = model.predict_proba(test[cols])[:, 1]
        pred = (proba > 0.5).astype(int)
        rows.append({"fusion(simulated)": fusion, "graph": graph,
                     "f1": round(f1_score(test["is_attacker"], pred), 4),
                     "roc_auc": round(roc_auc_score(test["is_attacker"], proba), 4),
                     "n_test": len(test)})
        if not a.compare:
            print(classification_report(test["is_attacker"], pred, digits=4))
            joblib.dump({"model": model, "features": cols}, "results/graph_xgb.joblib")
            test.to_parquet("results/graph_test.parquet")
    res = pd.DataFrame(rows)
    res["simulated"] = res["fusion(simulated)"]  # any run using sim_* features is simulated
    print(res.to_string(index=False))
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    res.to_csv("results/metrics/graph_variants.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "graph"], required=True)
    ap.add_argument("--sender-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-fusion", action="store_true", help="drop the simulated sim_* features")
    ap.add_argument("--no-graph", action="store_true", help="skip node2vec embeddings")
    ap.add_argument("--compare", action="store_true", help="graph variant: run all 4 ablations")
    a = ap.parse_args()
    if a.variant == "graph":
        return train_graph(a)
    df = load(a.sender_frac, a.seed)
    print(f"{len(df)} messages, {df['sender'].nunique()} senders")
    model, test = train_baseline(df, seed=a.seed)
    pred = model.predict(test[FEATURES])
    proba = model.predict_proba(test[FEATURES])[:, 1]
    print(classification_report(test["is_attacker"], pred, digits=4))
    print("ROC-AUC:", round(roc_auc_score(test["is_attacker"], proba), 4))
    print(pd.Series(model.feature_importances_, FEATURES).sort_values(ascending=False).round(3).to_string())
    Path("results").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, "results/baseline_rf.joblib")
    test.to_parquet("results/baseline_test.parquet")


if __name__ == "__main__":
    main()

"""Phase 8 ablation of the cooperative (SIMULATED) fusion features.

Every variant that keeps any sim_* column is a SIMULATED result (hidden-true-position
spoofing, src/spoof_sim.py). Deltas are F1(variant) - F1(full model), mean over `seeds`
sender-level train/test splits (seed changes the split and the classifier).

python -m src.ablation
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.classifiers.graph_gbdt import train_graph_classifier

CLAIM = ["sim_claim_fused_dist_med", "sim_claim_fused_dist_max"]
RESID = ["sim_resid_score_mean"]
VARIANTS = {  # name: (use_fusion, columns dropped)
    "full (all sim_*)": (True, []),
    "drop claim_fused_dist": (True, CLAIM),
    "drop residual score": (True, RESID),
    "drop both (keep n_obs/coverage only)": (True, CLAIM + RESID),
    "no fusion at all": (False, []),
}


def run(fv, emb=None, seeds=range(5)):
    rows = []
    for name, (fusion, drop) in VARIANTS.items():
        for seed in seeds:
            m, te, cols = train_graph_classifier(fv, emb, use_fusion=fusion, drop=drop, seed=seed)
            p = m.predict_proba(te[cols])[:, 1]
            rows.append({"variant": name, "seed": seed, "graph": emb is not None,
                         "f1": f1_score(te["is_attacker"], p >= 0.5),
                         "roc_auc": roc_auc_score(te["is_attacker"], p)})
    r = pd.DataFrame(rows)
    s = r.groupby("variant", sort=False)[["f1", "roc_auc"]].agg(["mean", "std"])
    s.columns = ["f1", "f1_std", "roc_auc", "roc_auc_std"]
    s = s.reset_index()
    full = s.loc[s["variant"] == "full (all sim_*)", "f1"].iloc[0]
    s["f1_delta_vs_full"] = s["f1"] - full
    s["graph"] = emb is not None
    s["simulated"] = s["variant"] != "no fusion at all"
    return s


def main():
    fv = pd.read_parquet("data/processed/features.parquet")
    emb = pd.read_parquet("data/processed/embeddings.parquet")
    out = pd.concat([run(fv, None), run(fv, emb)], ignore_index=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    out.to_csv("results/metrics/ablation.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

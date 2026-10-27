"""Build the demo's assets from the processed data (run once, needs data/processed/merged.parquet):

    python -m app.build_assets

Writes app/results.json (evaluation tables), app/model.joblib (a compact Random Forest, a few MB) and app/samples.json
(held-out senders with their true labels, for the demo's sample picker).
The model is the Phase 2 message-level baseline: measured, NOT using simulated fusion.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.classifiers.baseline_rf import FEATURES, train_baseline

OUT = Path(__file__).parent
MSG_COLS = ["timestamp", "x", "y", "speed", "heading", "acl", "pos_noise", "spd_noise"]
N_MSGS = 60  # messages kept per sample sender


def main(sender_frac=0.2, seed=42, n_per_group=14):
    df = pd.read_parquet("data/processed/merged.parquet")
    keep = np.random.default_rng(seed).choice(df["sender"].unique(),
                                              int(df["sender"].nunique() * sender_frac), replace=False)
    df = df[df["sender"].isin(keep)]
    # compact: few shallow-ish trees keep the file small enough to ship
    model, test = train_baseline(df, seed=seed, n_estimators=40, max_depth=14, min_samples_leaf=20)
    p = model.predict_proba(test[FEATURES])[:, 1]
    print("compact model on held-out senders: F1 %.4f  ROC-AUC %.4f" % (
        f1_score(test["is_attacker"], p >= 0.5), roc_auc_score(test["is_attacker"], p)))
    joblib.dump({"model": model, "features": FEATURES}, OUT / "model.joblib", compress=3)
    print("model.joblib: %.1f MB" % ((OUT / "model.joblib").stat().st_size / 1e6))

    # sample senders from the held-out (never trained on) set: genuine + attackers of varied classes
    rng = np.random.default_rng(seed)
    test_senders = test.groupby("sender").agg(attack_type=("attack_type", "max"), n=("sender", "size"))
    test_senders = test_senders[test_senders["n"] >= 30]
    picks = list(rng.choice(test_senders[test_senders.attack_type == 0].index, n_per_group, replace=False))
    attackers = test_senders[test_senders.attack_type > 0]
    for t in sorted(attackers.attack_type.unique()):  # one per attack class first
        picks.append(rng.choice(attackers[attackers.attack_type == t].index))
    samples = []
    for s in picks:
        g = df[df["sender"] == s].sort_values("timestamp").iloc[:N_MSGS]
        t = int(g["attack_type"].max())
        samples.append({"id": f"sender {int(s)}", "true_label": "attacker" if t else "genuine",
                        "attack_class": t, "messages": g[MSG_COLS].round(5).to_dict("records")})
    (OUT / "samples.json").write_text(json.dumps(samples))
    print(f"samples.json: {len(samples)} senders")


def build_results_json():
    """Bundle the evaluation tables the dashboard shows (run after src.evaluate / src.ablation)."""
    m = Path("results/metrics")
    ov = pd.read_csv(m / "comparison.csv")
    ab = pd.read_csv(m / "ablation.csv")
    pt = pd.read_csv(m / "per_attack_type.csv")
    sp = pd.concat([pd.read_csv(m / "simulated_fusion_30-300m.csv"), pd.read_csv(m / "simulated_fusion_10-50m.csv")])
    piv = pt.pivot(index="attack_type", columns="model", values="f1").round(3)
    out = {
        "comparison": ov[["model", "simulated", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "n", "n_attack"]].round(4).to_dict("records"),
        "ablation": ab[ab["graph"] == False][["variant", "f1", "f1_std", "f1_delta_vs_full", "simulated"]].round(4).to_dict("records"),
        "spoof": sp[["offset_m", "feature", "roc_auc"]].round(3).to_dict("records"),
        "per_class": [{"attack_type": int(k), **{c: float(v) for c, v in row.items()}} for k, row in piv.iterrows()],
        "dataset": {"messages": 3194808, "senders": 24663, "attack_share": 0.405, "attack_classes": 19, "windows": 59108},
    }
    (OUT / "results.json").write_text(json.dumps(out))
    print("results.json written")


if __name__ == "__main__":
    main()
    build_results_json()

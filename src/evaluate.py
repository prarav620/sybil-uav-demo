"""Evaluation: overall + per-attack-class metrics, ROC/PR curves, confusion matrices and a
side-by-side comparison of model variants (all at window level, identical test rows).

A variant that uses the sim_* fusion features is SIMULATED (hidden-true-position spoofing,
src/spoof_sim.py): it is flagged `simulated=True` in every CSV and drawn dashed with a
"(SIMULATED)" tag in every figure. Attack classes are the dataset's numeric `class` ids
(names not mapped). Per-class metrics treat that class's windows as positives and all
genuine windows as negatives.

python -m src.evaluate            # trains the 4 variants (cached embeddings) and evaluates
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,  # noqa: E402
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

METRICS_DIR, FIG_DIR, PRED_DIR = Path("results/metrics"), Path("results/figures"), Path("results/preds")
EMB_PATH = Path("data/processed/embeddings.parquet")
VARIANTS = {  # name: (use_fusion, use_graph)
    "features only": (False, False),
    "graph": (False, True),
    "fusion (SIMULATED)": (True, False),
    "graph + fusion (SIMULATED)": (True, True),
}


def is_simulated(name):
    return "SIMULATED" in name


def overall_metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
            "fpr": fp / max(fp + tn, 1), "n": len(y), "n_attack": int(y.sum())}


def per_attack_type(df, thr=0.5):
    """df: columns is_attacker, attack_type, proba. One row per attack class."""
    gen = df[df["is_attacker"] == 0]
    rows = []
    for t, grp in df[df["is_attacker"] == 1].groupby("attack_type"):
        sub = pd.concat([gen, grp])
        y, p = sub["is_attacker"].to_numpy(), sub["proba"].to_numpy()
        pred = (p >= thr).astype(int)
        rows.append({"attack_type": int(t), "n_windows": len(grp),
                     "recall": recall_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
                     "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p)})
    return pd.DataFrame(rows)


def evaluate_all(preds):
    """preds: {name: DataFrame[is_attacker, attack_type, proba]} -> (overall, per_type)."""
    ov, pt = [], []
    for name, d in preds.items():
        ov.append({"model": name, "simulated": is_simulated(name),
                   **overall_metrics(d["is_attacker"].to_numpy(), d["proba"].to_numpy())})
        t = per_attack_type(d)
        t.insert(0, "simulated", is_simulated(name))
        t.insert(0, "model", name)
        pt.append(t)
    return pd.DataFrame(ov), pd.concat(pt, ignore_index=True)


def plot_all(preds, per_type):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(preds, ["#4c78a8", "#59a14f", "#e15759", "#b07aa1"]))

    def style(n):
        return "--" if is_simulated(n) else "-"

    for kind, fn, xl, yl in [("roc", roc_curve, "False positive rate", "True positive rate"),
                             ("pr", precision_recall_curve, "Recall", "Precision")]:
        fig, ax = plt.subplots(figsize=(6, 5))
        for n, d in preds.items():
            a, b, _ = fn(d["is_attacker"], d["proba"])
            ax.plot(*((a, b) if kind == "roc" else (b, a)), style(n), color=colors[n], label=n)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ":", color="gray")
        ax.set(xlabel=xl, ylabel=yl, title=f"{kind.upper()} curves (dashed = SIMULATED fusion features)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{kind}_curves.png", dpi=130)
        plt.close(fig)
    fig, axs = plt.subplots(1, len(preds), figsize=(3.2 * len(preds), 3.3))
    for ax, (n, d) in zip(np.atleast_1d(axs), preds.items()):
        cm = confusion_matrix(d["is_attacker"], (d["proba"] >= 0.5).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center",
                    color="black" if v < cm.max() / 2 else "white", fontsize=9)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["genuine", "attack"],
               yticklabels=["genuine", "attack"], title=n.replace(" (SIMULATED)", "\n(SIMULATED)"))
        ax.title.set_fontsize(8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=130)
    plt.close(fig)
    piv = per_type.pivot(index="attack_type", columns="model", values="f1")[list(preds)]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(piv.columns)
    for i, n in enumerate(piv.columns):
        ax.bar(np.arange(len(piv)) + i * w, piv[n], w, color=colors[n],
               hatch="//" if is_simulated(n) else None, label=n)
    ax.set(xticks=np.arange(len(piv)) + 0.4 - w / 2, xticklabels=piv.index, xlabel="attack class id",
           ylabel="F1 (class vs genuine)",
           title="Per-attack-class F1 (hatched = SIMULATED fusion features)", ylim=(0.5, 1.02))
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_attack_f1.png", dpi=130)
    plt.close(fig)


def run_variants(seed=42):
    from src.classifiers.graph_gbdt import train_graph_classifier
    from src.graph_builder import build_graph, node2vec_embeddings
    fv = pd.read_parquet("data/processed/features.parquet")
    if EMB_PATH.exists():
        emb = pd.read_parquet(EMB_PATH)
    else:
        emb = node2vec_embeddings(build_graph(fv), seed=seed)
        emb.to_parquet(EMB_PATH)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    preds = {}
    for name, (fusion, graph) in VARIANTS.items():
        model, test, cols = train_graph_classifier(fv, emb if graph else None, use_fusion=fusion, seed=seed)
        d = test[["is_attacker", "attack_type", "sender"]].copy()
        d["proba"] = model.predict_proba(test[cols])[:, 1]
        fname = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "and")
        d.to_parquet(PRED_DIR / f"{fname}.parquet")
        preds[name] = d
    return preds


def main():
    preds = run_variants()
    ov, pt = evaluate_all(preds)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ov.to_csv(METRICS_DIR / "comparison.csv", index=False)
    pt.to_csv(METRICS_DIR / "per_attack_type.csv", index=False)
    plot_all(preds, pt)
    print(ov.round(4).to_string(index=False))
    print(pt.pivot(index="attack_type", columns="model", values="f1").round(3).to_string())


if __name__ == "__main__":
    main()

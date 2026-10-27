"""Feature vector (+ optional node2vec embedding) -> XGBoost, split by sender.

Fusion columns (sim_*) come from the hidden-true-position simulation: any metric of a
model trained with use_fusion=True is a SIMULATED result.
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

NON_FEATURES = {"sender", "window", "is_attacker", "attack_type", "x_mean", "y_mean"}


def feature_columns(fv, use_fusion=True, drop=()):
    cols = [c for c in fv.columns if c not in NON_FEATURES and c not in set(drop)]
    return [c for c in cols if use_fusion or not c.startswith("sim_")]


def train_graph_classifier(features_df, embeddings=None, use_fusion=True, test_size=0.3,
                           seed=42, drop=(), **xgb):
    """Return (model, test_df, columns). embeddings=None trains without the graph part.
    test_df holds the held-out rows' features plus labels, for senders unseen in training."""
    X = features_df[feature_columns(features_df, use_fusion, drop)]
    if embeddings is not None:
        X = pd.concat([X, embeddings.loc[features_df.index]], axis=1)
    y = features_df["is_attacker"].astype(int).to_numpy()
    tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed)
                  .split(X, y, groups=features_df["sender"]))
    params = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.8, n_jobs=-1, random_state=seed, eval_metric="logloss")
    params.update(xgb)
    model = XGBClassifier(**params).fit(X.iloc[tr], y[tr])
    test = X.iloc[te].copy()
    test["is_attacker"] = y[te]
    test["attack_type"] = features_df["attack_type"].to_numpy()[te]
    test["sender"] = features_df["sender"].to_numpy()[te]
    return model, test, list(X.columns)

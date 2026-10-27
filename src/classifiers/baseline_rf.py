"""Single-observer baseline (UAVSyDM-style), adapted to the single-table VeReMi data.

The Mendeley table has no receiver column, so each feature is a proxy computed from a
sender's own past messages (causal: only earlier messages are used):
  signal consistency   -> reported pos/speed noise, claimed vs implied speed gap
  DoA stability proxy  -> change of claimed heading and of implied motion direction
  broadcast reliability-> deviation of the inter-message gap from the nominal 1 s
  trust blend          -> direct (rolling plausibility of own recent messages) and
                          indirect (consistency of the sender's pseudonym neighbours
                          in time, i.e. pseudonym changes) blended 0.7 / 0.3
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

FEATURES = [
    "pos_noise", "spd_noise", "speed", "acl",
    "speed_gap", "implied_speed", "heading_change", "motion_dir_change",
    "dt", "dt_dev", "direct_trust", "indirect_trust", "trust",
]
NOMINAL_DT = 1.0
TRUST_WINDOW = 5


def build_baseline_features(df):
    df = df.sort_values(["sender", "timestamp"]).reset_index(drop=True)
    g = df.groupby("sender", sort=False)
    dt = g["timestamp"].diff()
    dx, dy = g["x"].diff(), g["y"].diff()
    safe_dt = dt.where(dt > 0)
    implied = np.hypot(dx, dy) / safe_dt
    f = pd.DataFrame(index=df.index)
    f["pos_noise"], f["spd_noise"] = df["pos_noise"], df["spd_noise"]
    f["speed"], f["acl"] = df["speed"], df["acl"]
    f["implied_speed"] = implied
    f["speed_gap"] = (implied - df["speed"]).abs()
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    f["heading_change"] = wrap(g["heading"].diff()).abs()
    f["motion_dir_change"] = wrap(np.arctan2(dy, dx) - df["heading"]).abs()
    f["dt"] = dt
    f["dt_dev"] = (dt - NOMINAL_DT).abs()
    # direct trust: share of recent own messages whose claimed speed matches the implied one
    ok = (f["speed_gap"] < 5.0).astype(float).where(f["speed_gap"].notna())
    f["direct_trust"] = ok.groupby(df["sender"]).transform(
        lambda s: s.shift(1).rolling(TRUST_WINDOW, min_periods=1).mean())
    # indirect trust: same measure aggregated over messages sharing the pseudonym
    f["indirect_trust"] = ok.groupby(df["sender_pseudo"]).transform(
        lambda s: s.shift(1).expanding().mean())
    f["trust"] = 0.7 * f["direct_trust"] + 0.3 * f["indirect_trust"]
    # first message of a sender has no history: neutral values
    f = f.fillna({"direct_trust": 0.5, "indirect_trust": 0.5, "trust": 0.5}).fillna(0.0)
    return df, f[FEATURES]


def train_baseline(df, test_size=0.3, seed=42, n_estimators=100, **rf_kwargs):
    """Return (model, test_df) where test_df holds features, label and attack_type
    for senders never seen in training."""
    df, X = build_baseline_features(df)
    y = df["is_attacker"].to_numpy()
    tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed)
                  .split(X, y, groups=df["sender"]))
    model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                   class_weight="balanced", random_state=seed, **rf_kwargs)
    model.fit(X.iloc[tr], y[tr])
    test = X.iloc[te].copy()
    test["is_attacker"] = y[te]
    test["attack_type"] = df["attack_type"].to_numpy()[te]
    test["sender"] = df["sender"].to_numpy()[te]
    return model, test

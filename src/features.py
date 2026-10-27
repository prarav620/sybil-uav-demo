"""Per-(vehicle, window) feature vector for classifier training.

Columns
  base_*      Phase 2 baseline features, window mean (+ max of speed_gap)
  speed_var, heading_change_rate, pos_var      sliding-window behavioural features
  sim_*       cooperative AoA-fusion features. SIMULATED: bearings come from a hidden
              true position (see src/spoof_sim.py), so every sim_* column and any result
              that depends on one must be reported as simulated.
  n_msgs      messages in the window (not simulated)
  x_mean, y_mean   window-mean claimed position (used for graph edges, not as classifier input)
  is_attacker, attack_type    labels (window is an attacker window if any message is)

Fusion is computed per 1 s sub-window, then aggregated to `window` seconds (an integer
number of seconds). Sub-windows without 2+ observers have no fusion; windows with no
fused sub-window keep NaN in the sim_* columns (tree models handle NaN) and
sim_n_fused = 0 says so explicitly.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.classifiers.baseline_rf import FEATURES as BASE_FEATURES, build_baseline_features
from src.spoof_sim import run as run_spoof_sim

CLAIM_DIST_CLIP = 500.0  # m; near-parallel bearings give wild fixes


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def behavioural(df, w):
    """speed variance, heading-change rate (rad/s) and position variance per window."""
    g = df.assign(w=w).groupby(["sender", "w"])
    hc = df.groupby("sender")["heading"].diff().pipe(_wrap).abs()
    dt = df.groupby("sender")["timestamp"].diff()
    out = pd.DataFrame({
        "speed_var": g["speed"].var(ddof=0),
        "pos_var": g["x"].var(ddof=0) + g["y"].var(ddof=0),
        "n_msgs": g.size(),
        "x_mean": g["x"].mean(),
        "y_mean": g["y"].mean(),
    })
    out["heading_change_rate"] = (hc / dt.where(dt > 0)).groupby([df["sender"], w]).mean()
    return out


def fusion_features(f1, window):
    """Aggregate 1 s fusion rows (output of spoof_sim.run) to `window` seconds."""
    f = f1.copy()
    f["w"] = f["window"] // int(window)
    f["cfd"] = f["claim_fused_dist"].clip(upper=CLAIM_DIST_CLIP)
    f["fused"] = ~f["single_observer"]
    f.loc[~f["fused"], "cfd"] = np.nan
    g = f.groupby(["sender", "w"])
    return pd.DataFrame({
        "sim_claim_fused_dist_med": g["cfd"].median(),
        "sim_claim_fused_dist_max": g["cfd"].max(),
        "sim_n_obs_mean": f["n_obs"].where(f["fused"]).groupby([f["sender"], f["w"]]).mean(),
        "sim_resid_score_mean": g["angular_consistency_score"].mean(),
        "sim_single_observer_frac": g["single_observer"].mean(),
        "sim_n_fused": g["fused"].sum(),
    })


def build_feature_vector(df, window=10, fusion=True, n_observers=15, coverage_radius=250.0,
                         offset_range=(30.0, 300.0), sigma_deg=2.0, seed=0):
    window = int(window)
    df, base = build_baseline_features(df)  # sorted by (sender, timestamp), reset index
    w = (df["timestamp"] // window).astype(int).rename("w")
    key = [df["sender"], w]
    out = base.groupby(key).mean().add_prefix("base_")
    out["base_speed_gap_max"] = base["speed_gap"].groupby(key).max()
    out = out.join(behavioural(df, w))
    out["is_attacker"] = df["is_attacker"].groupby(key).max()
    out["attack_type"] = df["attack_type"].groupby(key).max()
    if fusion:
        f1 = run_spoof_sim(df, n_observers, coverage_radius, offset_range, sigma_deg, seed=seed)
        out = out.join(fusion_features(f1, window))
    out = out.reset_index().rename(columns={"w": "window"})
    # a single message has no variance / heading change: genuinely zero, not missing
    for c in ("speed_var", "pos_var", "heading_change_rate"):
        out[c] = out[c].fillna(0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender-frac", type=float, default=0.2)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_parquet("data/processed/merged.parquet")
    s = np.random.default_rng(a.seed).choice(df["sender"].unique(),
        int(df["sender"].nunique() * a.sender_frac), replace=False)
    fv = build_feature_vector(df[df["sender"].isin(s)], a.window, seed=a.seed)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    fv.to_parquet("data/processed/features.parquet", index=False)
    print(f"{len(fv)} rows, attacker share {fv['is_attacker'].mean():.3f}")
    print("NaN share per column:\n", fv.isna().mean().round(3).to_string())
    full = fv["sim_n_fused"] >= 1
    print(f"windows with >=1 fused sub-window: {full.mean():.3f}")
    print("NaNs among those (excluding resid score):",
          int(fv.loc[full].drop(columns="sim_resid_score_mean").isna().sum().sum()))


if __name__ == "__main__":
    main()

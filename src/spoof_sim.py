"""SIMULATED spoofing evaluation of the cooperative AoA fusion.

Why this exists: the Mendeley table only contains *claimed* positions, so bearings computed
from them are always self-consistent and the angular-consistency score cannot show
anything about attackers. This module adds an explicit, synthetic assumption:

    genuine message : true position = claimed position
    attacker message: true position = claimed position - offset, where the offset has a
                      uniformly random direction and a magnitude drawn uniformly from
                      offset_range (metres). The physical transmitter sits at the true
                      position, so observers measure bearings to it.

Coverage rule: which UAVs hear a message is decided from the CLAIMED position (bearings
still point at the true one). Using the true position made n_obs leak the label, because
the offset moves attackers in and out of coverage (fixed in Phase 8b). This is slightly
unphysical but keeps coverage features label-independent.

Assumptions to keep in mind when reading results:
  * every attack class (or only `spoofed_classes`, if given) is treated as position
    spoofing; in reality some VeReMi attacks (e.g. speed or DoS types) do not falsify
    position, so this over-states how many attackers are geometrically detectable;
  * the offset is independent per message, bearings get Gaussian error `sigma_deg`;
  * detection numbers from this module are SIMULATED and must be labelled as such in
    any report; they say how well the method could work if attackers spoofed position by
    roughly this much, not how it performs on real VeReMi traffic.

python -m src.spoof_sim [--offset-min 30 --offset-max 300 --sigma-deg 2]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.doa_fusion import add_aoa_noise, fuse
from src.uav_overlay import assign_observers, place_observers


def add_true_positions(df, offset_range=(30.0, 300.0), spoofed_classes=None, seed=0):
    """Add tx, ty (hidden true position) and `spoofed` (bool) to a copy of df."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    spoofed = out["is_attacker"].to_numpy() == 1
    if spoofed_classes is not None:
        spoofed &= out["attack_type"].isin(spoofed_classes).to_numpy()
    mag = rng.uniform(*offset_range, len(out))
    ang = rng.uniform(0, 2 * np.pi, len(out))
    out["tx"] = out["x"] - spoofed * mag * np.cos(ang)
    out["ty"] = out["y"] - spoofed * mag * np.sin(ang)
    out["spoofed"] = spoofed
    return out


def run(df, n_observers=15, coverage_radius=250.0, offset_range=(30.0, 300.0),
        sigma_deg=2.0, spoofed_classes=None, seed=0):
    d = add_true_positions(df.reset_index(drop=True), offset_range, spoofed_classes, seed)
    obs = place_observers(d, n_observers, placement="kmeans", seed=seed)
    ld = assign_observers(d, observers=obs, coverage_radius=coverage_radius, xcol="tx", ycol="ty",
                         cover_xcol="x", cover_ycol="y")
    ld = add_aoa_noise(ld, sigma_deg, seed)
    f = fuse(d, ld, obs)
    lab = (d.assign(window=(d["timestamp"] // 1).astype(int))
             .groupby(["sender", "window"])["spoofed"].max().rename("is_spoofed").reset_index())
    return f.merge(lab, on=["sender", "window"])


def evaluate(f):
    rows = []
    for name, mask in [("angular_consistency_score", f["n_obs"] >= 3),
                       ("claim_fused_dist", f["n_obs"] >= 2)]:
        s = f[mask].dropna(subset=[name])
        y = s["is_spoofed"].astype(int)
        auc = roc_auc_score(y, s[name]) if y.nunique() == 2 else np.nan
        rows.append({"feature": name, "n_windows": len(s), "spoofed_share": round(y.mean(), 3),
                     "mean_genuine": s.loc[y == 0, name].mean(), "mean_spoofed": s.loc[y == 1, name].mean(),
                     "roc_auc": auc, "simulated": True})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender-frac", type=float, default=0.2)
    ap.add_argument("--offset-min", type=float, default=30.0)
    ap.add_argument("--offset-max", type=float, default=300.0)
    ap.add_argument("--sigma-deg", type=float, default=2.0)
    ap.add_argument("--n-observers", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_parquet("data/processed/merged.parquet")
    s = np.random.default_rng(a.seed).choice(df["sender"].unique(),
        int(df["sender"].nunique() * a.sender_frac), replace=False)
    df = df[df["sender"].isin(s)]
    f = run(df, a.n_observers, 250.0, (a.offset_min, a.offset_max), a.sigma_deg, seed=a.seed)
    res = evaluate(f)
    res.insert(0, "offset_m", f"{a.offset_min:g}-{a.offset_max:g}")
    print("SIMULATED RESULTS (hidden true position = claimed - random offset)")
    print(res.round(4).to_string(index=False))
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    res.to_csv(f"results/metrics/simulated_fusion_{a.offset_min:g}-{a.offset_max:g}m.csv", index=False)


if __name__ == "__main__":
    main()

"""Virtual multi-UAV overlay: place observers over the map and compute bearings.

Bearings are ideal (noise-free) geometric values from the sender's claimed position.
Conventions: azimuth = atan2(dy, dx) in radians (east = 0, counter-clockwise);
elevation = depression angle below the observer's horizontal plane, atan2(altitude, range)
(pi/2 when directly overhead). Senders are taken to be at z = 0.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

COVERAGE_RADIUS = 250.0  # m, horizontal (matches UAVSyDM)


def place_observers(df, n_observers=3, altitude=100.0, placement="kmeans", seed=0):
    """Return DataFrame [observer, ox, oy, oz].

    placement: 'grid' (centres of an even grid over the bounding box),
    'random' (uniform in the box) or 'kmeans' (cluster centres of sender positions,
    which puts UAVs where traffic actually is).
    """
    x0, x1, y0, y1 = df["x"].min(), df["x"].max(), df["y"].min(), df["y"].max()
    rng = np.random.default_rng(seed)
    if placement == "random":
        pts = np.column_stack([rng.uniform(x0, x1, n_observers), rng.uniform(y0, y1, n_observers)])
    elif placement == "grid":
        cols = int(np.ceil(np.sqrt(n_observers)))
        rows = int(np.ceil(n_observers / cols))
        gx = x0 + (np.arange(cols) + 0.5) * (x1 - x0) / cols
        gy = y0 + (np.arange(rows) + 0.5) * (y1 - y0) / rows
        pts = np.array([(a, b) for b in gy for a in gx])[:n_observers]
    elif placement == "kmeans":
        sample = df[["x", "y"]].sample(min(len(df), 200_000), random_state=seed)
        pts = KMeans(n_observers, n_init=3, random_state=seed).fit(sample).cluster_centers_
    else:
        raise ValueError(f"unknown placement {placement!r}")
    return pd.DataFrame({"observer": range(len(pts)), "ox": pts[:, 0], "oy": pts[:, 1], "oz": altitude})


def assign_observers(df, n_observers=3, altitude=100.0, coverage_radius=COVERAGE_RADIUS,
                     placement="kmeans", seed=0, observers=None, xcol="x", ycol="y",
                     cover_xcol=None, cover_ycol=None):
    """Long format: one row per (message, observer within coverage_radius).

    Columns: msg_idx (index into df), observer, azimuth, elevation, range_h.
    xcol/ycol name the position columns the bearings are computed from (default: claimed
    x/y; the spoofing simulation passes the hidden true position instead). cover_xcol/
    cover_ycol, if given, decide coverage (range test) from those columns instead.
    """
    obs = observers if observers is not None else place_observers(
        df, n_observers, altitude, placement, seed)
    x, y = df[xcol].to_numpy(), df[ycol].to_numpy()
    # which observers hear a message may be decided from different (claimed) coordinates
    # than the ones the bearings point at; this keeps coverage independent of any hidden
    # spoofing offset in the simulation (otherwise n_obs would leak the label)
    cx = df[cover_xcol].to_numpy() if cover_xcol else x
    cy = df[cover_ycol].to_numpy() if cover_ycol else y
    parts = []
    for o in obs.itertuples():
        dx, dy = x - o.ox, y - o.oy
        rng_h = np.hypot(dx, dy)
        m = np.flatnonzero(np.hypot(cx - o.ox, cy - o.oy) < coverage_radius)
        # bearing from observer to sender
        parts.append(pd.DataFrame({
            "msg_idx": df.index.to_numpy()[m],
            "observer": o.observer,
            "azimuth": np.arctan2(dy[m], dx[m]),
            "elevation": np.arctan2(o.oz, rng_h[m]),
            "range_h": rng_h[m],
        }))
    return pd.concat(parts, ignore_index=True)


def coverage_summary(df, long_df):
    """Share of messages seen by 0, 1, 2, ... observers."""
    n = long_df.groupby("msg_idx").size().reindex(df.index, fill_value=0)
    return n.value_counts(normalize=True).sort_index()


if __name__ == "__main__":
    d = pd.read_parquet("data/processed/merged.parquet", columns=["sender", "x", "y"])
    for pl in ("grid", "kmeans"):
        o = place_observers(d, 3, placement=pl)
        print(pl, o.round(0).to_dict("records"))
        print(coverage_summary(d, assign_observers(d, observers=o)).round(3).to_string())

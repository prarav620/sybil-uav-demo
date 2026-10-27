"""Cooperative AoA fusion across observers (sequential linearized LMMSE / EKF-style).

For one message, each observer i at o_i reports an azimuth theta_i to the transmitter.
The source position p is estimated by processing observers sequentially:
    h_i(p) = atan2(p_y - o_iy, p_x - o_ix),  H_i = [-dy/r^2, dx/r^2]
    S = H P H' + sigma^2,  K = P H'/S,  p += K * wrap(theta_i - h_i(p)),  P -= K H P
starting from a two-bearing intersection with a wide prior. The pass is repeated once,
re-linearising at the previous estimate (iterated update). The residual of observer i is
wrap(theta_i - h_i(p_fused)); angular_consistency_score summarises residuals per
(vehicle, time window). Fusion is horizontal (azimuth only).
"""
import numpy as np
import pandas as pd

PRIOR_STD = 200.0  # m


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def add_aoa_noise(long_df, sigma_deg=2.0, seed=0):
    """Return a copy of the long bearings frame with Gaussian azimuth error."""
    out = long_df.copy()
    out["azimuth"] = wrap(out["azimuth"] + np.random.default_rng(seed).normal(
        0, np.deg2rad(sigma_deg), len(out)))
    return out


def fuse_bearings(ox, oy, theta, sigma):
    """Fuse padded arrays of shape (M, K) (NaN where an observer is absent; each row
    has >= 2 valid entries, valid ones first). Returns fused (M,2), residuals (M,K)."""
    M, K = theta.shape
    u1 = np.column_stack([np.cos(theta[:, 0]), np.sin(theta[:, 0])])
    u2 = np.column_stack([np.cos(theta[:, 1]), np.sin(theta[:, 1])])
    d = np.column_stack([ox[:, 1] - ox[:, 0], oy[:, 1] - oy[:, 0]])
    cr = u1[:, 0] * u2[:, 1] - u1[:, 1] * u2[:, 0]
    t1 = (d[:, 0] * u2[:, 1] - d[:, 1] * u2[:, 0]) / np.where(np.abs(cr) < 1e-3, np.nan, cr)
    p0 = np.column_stack([ox[:, 0], oy[:, 0]]) + t1[:, None] * u1
    # near-parallel bearings: fall back to the first observer's ray at the prior range
    p0 = np.where(np.isfinite(p0), p0,
                  np.column_stack([ox[:, 0], oy[:, 0]]) + PRIOR_STD * u1)
    p = p0.copy()
    for _ in range(2):
        P = np.tile(np.eye(2) * PRIOR_STD**2, (M, 1, 1))
        est = p0.copy() if _ == 0 else p.copy()
        lin = est.copy()  # linearisation point, updated as the estimate improves
        for k in range(K):
            ok = ~np.isnan(theta[:, k])
            dx, dy = lin[:, 0] - ox[:, k], lin[:, 1] - oy[:, k]
            r2 = np.maximum(dx**2 + dy**2, 1.0)
            H = np.column_stack([-dy / r2, dx / r2])                       # (M,2)
            PH = np.einsum("mij,mj->mi", P, H)
            S = np.einsum("mi,mi->m", H, PH) + sigma**2
            Kg = PH / S[:, None]
            innov = wrap(theta[:, k] - np.arctan2(est[:, 1] - oy[:, k], est[:, 0] - ox[:, k]))
            Kg = np.where(ok[:, None], Kg, 0.0)  # absent slot holds NaN: mask, don't multiply
            est = est + Kg * np.nan_to_num(innov)[:, None]
            P = P - np.einsum("mi,mj->mij", Kg, np.nan_to_num(PH))
            lin = est
        p = est
    resid = wrap(theta - np.arctan2(p[:, 1:2] - oy, p[:, 0:1] - ox))
    return p, resid


def fuse(df, long_df, observers, window=1.0, sigma_deg=2.0):
    """Per-(sender, window) fused position and angular_consistency_score.

    df: messages (sender, timestamp, x, y); long_df: output of assign_observers (use
    add_aoa_noise on it for a realistic measurement); observers: place_observers output.
    Rows with < 2 observers are kept with single_observer=True and no fusion; the score is
    NaN unless n_obs >= 3 (with 2 bearings the residual is exactly zero).
    """
    sigma = np.deg2rad(sigma_deg)
    ld = long_df.merge(observers[["observer", "ox", "oy"]], on="observer").sort_values(
        ["msg_idx", "observer"])
    ld["rank"] = ld.groupby("msg_idx").cumcount()
    n = ld.groupby("msg_idx").size()
    ld = ld[ld["msg_idx"].isin(n.index[n >= 2])]
    msgs = ld["msg_idx"].unique()
    K = int(ld["rank"].max()) + 1 if len(ld) else 2
    pos = pd.Series(np.arange(len(msgs)), index=msgs)
    r, c = pos.loc[ld["msg_idx"]].to_numpy(), ld["rank"].to_numpy()
    grid = lambda col: _scatter(r, c, ld[col].to_numpy(), len(msgs), K)
    p, resid = fuse_bearings(grid("ox"), grid("oy"), grid("azimuth"), sigma)
    m = pd.DataFrame({"msg_idx": msgs, "fx": p[:, 0], "fy": p[:, 1],
                      "rms": np.sqrt(np.nanmean(resid**2, axis=1)),
                      "maxabs": np.nanmax(np.abs(resid), axis=1), "n": (~np.isnan(resid)).sum(1)})
    base = pd.DataFrame({"msg_idx": df.index, "sender": df["sender"].to_numpy(),
                         "window": (df["timestamp"] // window).astype(int).to_numpy(),
                         "cx": df["x"].to_numpy(), "cy": df["y"].to_numpy()})
    base = base.merge(m, on="msg_idx", how="left")
    base["claim_dist"] = np.hypot(base["fx"] - base["cx"], base["fy"] - base["cy"])
    g = base.groupby(["sender", "window"])
    out = g.agg(fused_x=("fx", "mean"), fused_y=("fy", "mean"),
                angular_consistency_score=("rms", "mean"),
                max_residual=("maxabs", "max"),
                claim_fused_dist=("claim_dist", "mean"),
                n_obs=("n", "max")).reset_index()
    out["n_obs"] = out["n_obs"].fillna(0).astype(int)
    out["single_observer"] = out["n_obs"] < 2
    # 2 bearings fit the 2 unknowns exactly (residual 0 by construction): the consistency
    # score only carries information with >= 3 observers.
    out.loc[out["n_obs"] < 3, ["angular_consistency_score", "max_residual"]] = np.nan
    return out


def _scatter(r, c, vals, M, K):
    a = np.full((M, K), np.nan)
    a[r, c] = vals
    return a

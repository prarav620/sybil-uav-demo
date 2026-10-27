import numpy as np
import pandas as pd

from src.doa_fusion import fuse, fuse_bearings, wrap
from src.uav_overlay import assign_observers

OBS = pd.DataFrame({"observer": [0, 1, 2], "ox": [0.0, 400.0, 200.0],
                    "oy": [0.0, 0.0, 300.0], "oz": 100.0})


def _theta(true, k):
    return np.arctan2(true[1] - OBS.oy.to_numpy()[:k], true[0] - OBS.ox.to_numpy()[:k])


def test_fused_estimate_close_to_truth():
    true = np.array([180.0, 120.0])
    rng = np.random.default_rng(0)
    th = (_theta(true, 3) + rng.normal(0, np.deg2rad(1.0), 3))[None, :]
    p, res = fuse_bearings(OBS.ox.to_numpy()[None], OBS.oy.to_numpy()[None], th, np.deg2rad(1.0))
    assert np.hypot(*(p[0] - true)) < 10.0
    assert np.sqrt(np.mean(res**2)) < np.deg2rad(3)


def test_perturbed_bearing_scores_higher():
    true = np.array([180.0, 120.0])
    good = _theta(true, 3)[None, :]
    bad = good.copy()
    bad[0, 2] = wrap(bad[0, 2] + np.deg2rad(25))
    args = (OBS.ox.to_numpy()[None], OBS.oy.to_numpy()[None])
    _, rg = fuse_bearings(*args, good, np.deg2rad(1.0))
    _, rb = fuse_bearings(*args, bad, np.deg2rad(1.0))
    assert np.abs(rb).max() > 10 * np.abs(rg).max() + np.deg2rad(2)


def test_single_observer_flagged():
    df = pd.DataFrame({"sender": [1, 2], "timestamp": [0.0, 0.0],
                       "x": [180.0, 5000.0], "y": [120.0, 5000.0]})
    # sender 1 covered by all three observers, sender 2 by none (out of range)
    ld = assign_observers(df, observers=OBS, coverage_radius=450)
    out = fuse(df, ld, OBS).set_index("sender")
    assert not out.loc[1, "single_observer"] and out.loc[1, "claim_fused_dist"] < 1.0
    assert out.loc[2, "single_observer"] and np.isnan(out.loc[2, "angular_consistency_score"])


def test_ragged_rows_not_nan():
    true = np.array([180.0, 120.0])
    ox, oy = OBS.ox.to_numpy(), OBS.oy.to_numpy()
    th = np.full((2, 3), np.nan)
    th[0, :] = _theta(true, 3)
    th[1, :2] = _theta(true, 2)  # second row only has 2 observers
    X, Y = np.tile(ox, (2, 1)), np.tile(oy, (2, 1))
    X[1, 2] = Y[1, 2] = np.nan
    p, res = fuse_bearings(X, Y, th, np.deg2rad(1.0))
    assert np.isfinite(p).all() and np.hypot(*(p[1] - true)) < 1.0

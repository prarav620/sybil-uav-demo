import numpy as np
import pandas as pd

from src.uav_overlay import assign_observers, place_observers


def _obs(ox, oy, oz=100.0):
    return pd.DataFrame({"observer": [0], "ox": [ox], "oy": [oy], "oz": [oz]})


def test_known_geometry():
    df = pd.DataFrame({"x": [100.0, 0.0, 0.0, 400.0], "y": [0.0, 100.0, 0.0, 0.0]})
    out = assign_observers(df, observers=_obs(0.0, 0.0))
    assert set(out.msg_idx) == {0, 1, 2}  # the 400 m one is out of range
    r = out.set_index("msg_idx")
    assert np.isclose(r.loc[0, "azimuth"], 0.0)               # due east
    assert np.isclose(r.loc[1, "azimuth"], np.pi / 2)         # due north
    assert np.isclose(r.loc[0, "elevation"], np.pi / 4)       # 100 m out, 100 m up
    assert np.isclose(r.loc[2, "elevation"], np.pi / 2)       # directly below the UAV


def test_placements_in_box():
    df = pd.DataFrame({"x": np.random.rand(500) * 1000, "y": np.random.rand(500) * 1000})
    for pl in ("grid", "random", "kmeans"):
        o = place_observers(df, 3, placement=pl)
        assert len(o) == 3 and o.ox.between(0, 1000).all() and o.oy.between(0, 1000).all()

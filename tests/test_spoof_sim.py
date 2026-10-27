import numpy as np
import pandas as pd

from src.spoof_sim import add_true_positions


def test_offsets_only_for_attackers():
    df = pd.DataFrame({"x": np.arange(100.0), "y": np.zeros(100),
                       "is_attacker": [0, 1] * 50, "attack_type": [0, 3] * 50})
    d = add_true_positions(df, (30.0, 300.0), seed=1)
    off = np.hypot(d.x - d.tx, d.y - d.ty)
    assert (off[d.is_attacker == 0] == 0).all()
    assert off[d.is_attacker == 1].between(30, 300).all()
    d2 = add_true_positions(df, spoofed_classes=[99])
    assert not d2.spoofed.any() and (d2.x == d2.tx).all()


def test_coverage_independent_of_offset():
    from src.uav_overlay import assign_observers
    obs = pd.DataFrame({"observer": [0], "ox": [0.0], "oy": [0.0], "oz": 100.0})
    df = pd.DataFrame({"x": [100.0, 100.0], "y": [0.0, 0.0], "tx": [100.0, 900.0], "ty": [0.0, 0.0]})
    out = assign_observers(df, observers=obs, xcol="tx", ycol="ty", cover_xcol="x", cover_ycol="y")
    assert sorted(out.msg_idx) == [0, 1]  # both claimed in range, though one is truly 900 m away

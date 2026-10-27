import pandas as pd

from src.classifiers.baseline_rf import FEATURES, build_baseline_features


def _df():
    rows = []
    for t in range(5):  # genuine: moves 10 m/s as claimed
        rows.append(dict(sender=1, sender_pseudo=11, timestamp=float(t), x=10.0 * t, y=0.0,
                         speed=10.0, heading=0.0, acl=0.0, pos_noise=1.0, spd_noise=0.0,
                         attack_type=0, is_attacker=0))
    for t in range(5):  # claims 10 m/s but jumps 100 m per step
        rows.append(dict(sender=2, sender_pseudo=22, timestamp=float(t), x=100.0 * t, y=0.0,
                         speed=10.0, heading=0.0, acl=0.0, pos_noise=1.0, spd_noise=0.0,
                         attack_type=3, is_attacker=1))
    return pd.DataFrame(rows)


def test_features_shape_and_signal():
    df, X = build_baseline_features(_df())
    assert list(X.columns) == FEATURES and not X.isna().any().any()
    gap = X.groupby(df["sender"])["speed_gap"].max()
    assert gap[1] < 1e-9 and gap[2] > 50


def test_build_feature_vector_small():
    import numpy as np
    from src.features import build_feature_vector
    rng = np.random.default_rng(0)
    rows = []
    for s in range(20):
        for t in range(30):
            rows.append(dict(sender=s, sender_pseudo=s, timestamp=float(t), x=300 + 10.0 * t,
                             y=300 + 5.0 * s, speed=10.0, heading=0.0, acl=0.0, pos_noise=1.0,
                             spd_noise=0.0, attack_type=int(s % 4 == 0) * 3, is_attacker=int(s % 4 == 0)))
    fv = build_feature_vector(pd.DataFrame(rows), window=10, n_observers=4)
    assert len(fv) == 20 * 3
    assert {"sim_claim_fused_dist_med", "speed_var", "base_trust", "is_attacker"} <= set(fv.columns)
    assert not fv[["speed_var", "pos_var", "heading_change_rate"]].isna().any().any()

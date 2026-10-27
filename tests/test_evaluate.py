import numpy as np
import pandas as pd

from src.evaluate import evaluate_all, is_simulated, overall_metrics, per_attack_type


def _d():
    y = np.array([0] * 50 + [1] * 25 + [1] * 25)
    t = np.array([0] * 50 + [3] * 25 + [7] * 25)
    p = np.where(y == 1, 0.9, 0.1)
    p[-5:] = 0.2  # five missed class-7 windows
    return pd.DataFrame({"is_attacker": y, "attack_type": t, "proba": p})


def test_overall_and_per_type():
    d = _d()
    m = overall_metrics(d.is_attacker.to_numpy(), d.proba.to_numpy())
    assert m["fpr"] == 0 and abs(m["recall"] - 0.90) < 1e-9
    t = per_attack_type(d).set_index("attack_type")
    assert t.loc[3, "recall"] == 1.0 and abs(t.loc[7, "recall"] - 0.8) < 1e-9


def test_simulated_flag():
    ov, pt = evaluate_all({"a": _d(), "b (SIMULATED)": _d()})
    assert ov.set_index("model")["simulated"].to_dict() == {"a": False, "b (SIMULATED)": True}
    assert pt[pt.model == "b (SIMULATED)"].simulated.all() and is_simulated("x (SIMULATED)")

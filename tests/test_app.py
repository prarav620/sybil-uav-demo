import math

from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)


def _trace(jump):
    """10 messages claiming 10 m/s eastward; `jump` metres of extra displacement per step."""
    return [{"timestamp": float(t), "x": (10.0 + jump) * t, "y": 0.0, "speed": 10.0,
             "heading": 0.0, "acl": 0.0, "pos_noise": 1.0, "spd_noise": 0.0} for t in range(10)]


def test_health_and_samples():
    assert client.get("/api/health").json()["status"] == "ok"
    s = client.get("/api/samples").json()
    assert len(s) >= 20 and {"genuine", "attacker"} <= {x["true_label"] for x in s}
    assert len(client.get("/api/samples/0").json()["messages"]) >= 30
    assert client.get("/api/samples/9999").status_code == 404


def test_predict_shape_and_ordering():
    r = client.post("/api/predict", json={"messages": _trace(0.0)}).json()
    assert r["verdict"] in ("genuine", "attacker") and len(r["messages"]) == 10
    assert all(0 <= m["probability"] <= 1 for m in r["messages"])
    assert not math.isnan(r["mean_probability"])


def test_inconsistent_trace_scores_higher_than_consistent():
    good = client.post("/api/predict", json={"messages": _trace(0.0)}).json()["mean_probability"]
    bad = client.post("/api/predict", json={"messages": _trace(150.0)}).json()["mean_probability"]
    assert bad > good  # claims 10 m/s but teleports 160 m per second


def test_validation():
    assert client.post("/api/predict", json={"messages": _trace(0.0)[:1]}).status_code == 422
    assert client.post("/api/predict", json={"messages": [{"x": 1}]}).status_code == 422


def test_known_samples_mostly_correct():
    # held-out samples: the demo model should get most of them right
    hits = 0
    s = client.get("/api/samples").json()
    for x in s:
        r = client.post("/api/predict", json={"messages": client.get(f"/api/samples/{x['index']}").json()["messages"]}).json()
        hits += r["verdict"] == x["true_label"]
    assert hits / len(s) >= 0.8


def test_warnings_for_short_or_missing_noise():
    short = [{"timestamp": float(t), "x": 10.0 * t, "y": 0.0, "speed": 10.0, "heading": 0.0} for t in range(4)]
    r = client.post("/api/predict", json={"messages": short}).json()
    assert len(r["warnings"]) == 2  # missing noise columns + too few messages
    ok = client.post("/api/predict", json={"messages": _trace(0.0)}).json()
    assert ok["warnings"] == []  # 10 messages with noise supplied: nothing to warn about


def test_results_and_model_endpoints():
    r = client.get("/api/results").json()
    assert {"comparison", "ablation", "spoof", "per_class", "dataset"} <= set(r)
    assert len(r["comparison"]) == 4 and any(x["simulated"] for x in r["comparison"])
    assert len(r["per_class"]) == 19
    m = client.get("/api/model").json()
    assert m["n_trees"] > 0 and m["features"][0]["importance"] >= m["features"][-1]["importance"]


def test_predict_returns_explanation_fields():
    msgs = client.get("/api/samples/0").json()["messages"]
    r = client.post("/api/predict", json={"messages": msgs}).json()
    assert {"claimed_speed", "implied_speed", "speed_gap", "probability"} <= set(r["messages"][0])


def test_dashboard_page_has_tabs():
    html = client.get("/").text
    assert all(t in html for t in ("Live detector", "Drone simulator", "Results", "Pipeline", "Limits", "SIMULATED"))

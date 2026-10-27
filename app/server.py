"""Live demo: score one vehicle's BSM trace with the Phase 2 baseline Random Forest.

    uvicorn app.server:app --port 8000

The model is the measured baseline (sender-history features), not the simulated fusion
variant. Input must be messages from ONE sender, because the features compare each message
with that sender's earlier ones.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.classifiers.baseline_rf import build_baseline_features

HERE = Path(__file__).parent
MAX_MESSAGES = 2000
THRESHOLD = 0.5
MIN_RELIABLE = 10

_bundle = joblib.load(HERE / "model.joblib")
MODEL, FEATURES = _bundle["model"], _bundle["features"]
SAMPLES = json.loads((HERE / "samples.json").read_text())
RESULTS = json.loads((HERE / "results.json").read_text())

# plain-language names for the features the model uses
FEATURE_LABELS = {
    "pos_noise": "Reported position noise", "spd_noise": "Reported speed noise", "speed": "Claimed speed",
    "acl": "Claimed acceleration", "speed_gap": "Gap: claimed vs actual speed",
    "implied_speed": "Speed implied by movement", "heading_change": "Heading change between messages",
    "motion_dir_change": "Direction of motion vs claimed heading", "dt": "Time since previous message",
    "dt_dev": "Deviation from the 1 s message rhythm", "direct_trust": "Trust from own recent behaviour",
    "indirect_trust": "Trust from the pseudonym's history", "trust": "Combined trust",
}

app = FastAPI(title="VeReMi misbehaviour detector (demo)")


class Message(BaseModel):
    timestamp: float
    x: float
    y: float
    speed: float = Field(ge=0)
    heading: float  # radians
    acl: float = 0.0
    pos_noise: float | None = None  # reported position noise (m); genuine VeReMi traffic ~5-6
    spd_noise: float | None = None


class Trace(BaseModel):
    messages: list[Message] = Field(min_length=2, max_length=MAX_MESSAGES)


def score(messages):
    df = pd.DataFrame([m.model_dump() for m in messages])
    warnings = []
    if df[["pos_noise", "spd_noise"]].isna().any().any():
        warnings.append("pos_noise / spd_noise missing: treated as 0, which is rare for genuine traffic "
                        "in the training data (about 5-6 m position noise), so the verdict may lean "
                        "towards 'attacker'.")
        df[["pos_noise", "spd_noise"]] = df[["pos_noise", "spd_noise"]].fillna(0.0)
    if len(df) < MIN_RELIABLE:
        warnings.append(f"Only {len(df)} messages: the features compare each message with earlier ones, "
                        f"so traces under {MIN_RELIABLE} messages give unreliable verdicts.")
    df["sender"] = 1
    df["sender_pseudo"] = 1
    df["attack_type"] = 0
    df["is_attacker"] = 0
    df, X = build_baseline_features(df)
    p = MODEL.predict_proba(X[FEATURES])[:, 1]
    mean_p = float(p.mean())
    return {
        "warnings": warnings,
        "verdict": "attacker" if mean_p >= THRESHOLD else "genuine",
        "mean_probability": round(mean_p, 4),
        "flagged_fraction": round(float((p >= THRESHOLD).mean()), 4),
        "messages": [{"timestamp": float(t), "x": float(x), "y": float(y), "probability": round(float(q), 4),
                      "claimed_speed": round(float(cs), 3), "implied_speed": round(float(iv), 3), "speed_gap": round(float(g), 3)}
                     for t, x, y, q, cs, iv, g in zip(df["timestamp"], df["x"], df["y"], p, X["speed"],
                                                      X["implied_speed"], X["speed_gap"])],
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "samples": len(SAMPLES)}


@app.get("/api/samples")
def samples():
    return [{"index": i, "id": s["id"], "n_messages": len(s["messages"]),
             "true_label": s["true_label"], "attack_class": s["attack_class"]}
            for i, s in enumerate(SAMPLES)]


@app.get("/api/samples/{i}")
def sample(i: int):
    if not 0 <= i < len(SAMPLES):
        raise HTTPException(404, "no such sample")
    return SAMPLES[i]


@app.get("/api/results")
def results():
    return RESULTS


@app.get("/api/model")
def model_info():
    imp = sorted(zip(FEATURES, MODEL.feature_importances_), key=lambda t: -t[1])
    return {"type": "Random Forest", "n_trees": len(MODEL.estimators_), "features": [
        {"name": n, "label": FEATURE_LABELS.get(n, n), "importance": round(float(v), 4)} for n, v in imp]}


@app.post("/api/predict")
def predict(trace: Trace):
    return score(trace.messages)


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")

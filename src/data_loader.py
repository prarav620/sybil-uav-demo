"""Parse VeReMi Extension logs into one row per received BSM.

Expected layout (kept as downloaded, one folder per scenario):
    data/raw/<scenario>/traceJSON-<rcv>-<n>-A<attackFlag>-<start>-<id>.json
    data/raw/<scenario>/traceGroundTruthJSON-<...>.json

Receiver logs hold JSON lines: type 2 = receiver's own GPS, type 3 = received BSM
(sender, messageID, pos, spd, hed, ...). The ground-truth file holds one JSON line
per sent message with `messageID` and `attackerType` (0 = genuine).
Field names are read defensively since releases differ slightly; check with the
__main__ block on one scenario before trusting a full run.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
MIXALL_CSV = RAW_DIR / "mix" / "mixalldata_clean.csv"
OUT_PATH = Path("data/processed/merged.parquet")
_TRACE_RE = re.compile(r"traceJSON-(?P<rcv>\d+)-")


def _norm(v):
    """Speed/heading may be vectors [x, y, z]; return a list of 3 floats."""
    return list(v) + [0.0] * (3 - len(v)) if isinstance(v, (list, tuple)) else [v, 0.0, 0.0]


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_ground_truth(scenario_dir):
    """Map messageID -> attackerType over all ground-truth files in a scenario."""
    gt = {}
    for p in Path(scenario_dir).glob("traceGroundTruthJSON*.json"):
        for rec in _read_lines(p):
            if "messageID" in rec:
                gt[rec["messageID"]] = int(rec.get("attackerType", 0))
    return gt


def load_scenario(scenario_dir):
    """Return a DataFrame of received messages for one scenario folder."""
    scenario_dir = Path(scenario_dir)
    gt = load_ground_truth(scenario_dir)
    rows = []
    for p in scenario_dir.glob("traceJSON-*.json"):
        m = _TRACE_RE.match(p.name)
        receiver = int(m.group("rcv")) if m else -1
        for rec in _read_lines(p):
            if rec.get("type") != 3:
                continue
            pos, spd, hed = _norm(rec["pos"]), _norm(rec.get("spd", 0.0)), _norm(rec.get("hed", 0.0))
            mid = rec.get("messageID")
            rows.append({
                "scenario": scenario_dir.name,
                "sender": int(rec["sender"]),
                "sender_pseudo": rec.get("senderPseudo"),
                "receiver": receiver,
                "message_id": mid,
                "timestamp": float(rec["rcvTime"]),
                "send_time": float(rec.get("sendTime", rec["rcvTime"])),
                "x": pos[0], "y": pos[1],
                "speed": float(np.hypot(spd[0], spd[1])),
                "heading": float(np.arctan2(hed[1], hed[0])) if isinstance(rec.get("hed"), list) else float(hed[0]),
                "attack_type": gt.get(mid, 0),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["is_attacker"] = (df["attack_type"] > 0).astype(int)
    return df


def load_mixall(path=MIXALL_CSV, nrows=None):
    """Load the Mendeley `mixalldata_clean.csv` (one row per message, no receiver).

    `class` 0 = genuine, 1..19 = attack type. Returns the same core columns as
    load_scenario (receiver is absent, so it is NaN) plus the *_n noise columns.
    """
    d = pd.read_csv(path, nrows=nrows)
    out = pd.DataFrame({
        "scenario": "mixall",
        "sender": d["sender"],
        "sender_pseudo": d["senderPseudo"],
        "receiver": -1,
        "message_id": d["messageID"],
        "timestamp": d["sendTime"],
        "send_time": d["sendTime"],
        "x": d["posx"], "y": d["posy"],
        "speed": np.hypot(d["spdx"], d["spdy"]),
        "heading": np.arctan2(d["hedy"], d["hedx"]),
        "acl": np.hypot(d["aclx"], d["acly"]),
        "pos_noise": np.hypot(d["posx_n"], d["posy_n"]),
        "spd_noise": np.hypot(d["spdx_n"], d["spdy_n"]),
        "attack_type": d["class"].astype("int16"),
    })
    out["is_attacker"] = (out["attack_type"] > 0).astype("int8")
    return out


def find_scenarios(raw_dir=RAW_DIR):
    """Every folder under raw_dir that directly contains traceJSON files."""
    raw_dir = Path(raw_dir)
    return sorted({p.parent for p in raw_dir.rglob("traceJSON-*.json")})


def load_all(raw_dir=RAW_DIR):
    frames = [load_scenario(s) for s in find_scenarios(raw_dir)]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", help="single scenario folder (sanity check)")
    ap.add_argument("--raw", default=str(RAW_DIR))
    ap.add_argument("--mixall", action="store_true", help="use mix/mixalldata_clean.csv (Mendeley release)")
    ap.add_argument("--all", action="store_true", help="parse every scenario and write merged.parquet")
    a = ap.parse_args()
    if a.mixall:
        df = load_mixall(nrows=None if a.all else 100000)
        if not a.all:
            df.info(); print(df.head()); print("attacker ratio:", df["is_attacker"].mean()); return
    elif a.all:
        df = load_all(a.raw)
    else:
        df = None
    if df is not None:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_PATH, index=False)
        print(f"wrote {OUT_PATH}: {len(df)} rows, attacker ratio {df['is_attacker'].mean():.3f}")
        return
    scen = a.scenario or (find_scenarios(a.raw) or [None])[0]
    if scen is None:
        raise SystemExit(f"No traceJSON files under {a.raw}; see data/README.md")
    df = load_scenario(scen)
    df.info()
    print(df.head())
    print("attacker ratio:", df["is_attacker"].mean())


if __name__ == "__main__":
    main()

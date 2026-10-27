# Cooperative multi-UAV misbehaviour detection on VeReMi Extension

**Live demo:** https://sybil-uav-demo.onrender.com (free tier; first load may take ~30-60 s) | **Code:** https://github.com/prarav620/sybil-uav-demo

Single-observer-style baseline, a cooperative multi-UAV AoA fusion layer, a
spatiotemporal graph classifier, and an evaluation/ablation framework, built on the
public VeReMi Extension dataset. Full write-up: [docs/REPORT.md](docs/REPORT.md).

**Important:** the dataset has only *claimed* positions. Everything that uses the
cooperative fusion features (`sim_*`) rests on a **simulated** hidden-true-position
spoofing model (`src/spoof_sim.py`) and is labelled SIMULATED everywhere. Those numbers
are not measured VeReMi results.

## Reproduce from a clean clone
Python 3.10+, about 5 GB free disk. Commands are for Windows Git Bash; on Linux/macOS use
`venv/bin/python` instead of `venv/Scripts/python`.

```bash
python -m venv venv
venv/Scripts/python -m pip install -r requirements.txt

# 1. Data: download "VeReMi_Extension Dataset for Misbehaviors in VANETs" (314 MB zip) from
#    https://data.mendeley.com/datasets/k62n4z9gdz/1 , extract the zip, then extract the inner
#    mixalldata_clean.tar.xz so that this file exists:
#      data/raw/mix/mixalldata_clean.csv        (~1.2 GB)
venv/Scripts/python -m src.data_loader --mixall --all     # -> data/processed/merged.parquet

# 2. Phase 2 message-level baseline (not simulated)         -> results/baseline_rf.joblib
venv/Scripts/python src/train.py --variant baseline

# 3. Window features incl. simulated fusion features         -> data/processed/features.parquet
venv/Scripts/python -m src.features

# 4. Simulated spoof-detection check (optional)              -> results/metrics/simulated_fusion_*.csv
venv/Scripts/python -m src.spoof_sim --offset-min 30 --offset-max 300
venv/Scripts/python -m src.spoof_sim --offset-min 10 --offset-max 50

# 5. Evaluation of the four window-level variants (~10 min first run: node2vec)
#    -> results/metrics/comparison.csv, per_attack_type.csv, results/figures/*.png
venv/Scripts/python -m src.evaluate

# 6. Ablation of the simulated fusion features              -> results/metrics/ablation.csv
venv/Scripts/python -m src.ablation

venv/Scripts/python -m pytest -q                            # 16 tests
```

All experiments use 20% of senders (whole senders sampled, seed 0 for sampling, 42 for the
train/test split) to stay CPU-friendly; pass `--sender-frac 1.0` to `src.features` and
`src/train.py` to use everything. `data/raw`, `data/processed` and `results/figures/*.png`
are gitignored. `src.evaluate` caches node2vec embeddings in
`data/processed/embeddings.parquet`; delete that file whenever `features.parquet` is rebuilt.

## Layout
| file | role |
|---|---|
| `src/data_loader.py` | parses the Mendeley CSV (and the original per-vehicle JSON logs) |
| `src/classifiers/baseline_rf.py`, `src/train.py` | Phase 2 baseline (sender-history proxy features) |
| `src/uav_overlay.py` | virtual UAV placement and ideal azimuth/elevation bearings |
| `src/doa_fusion.py` | sequential linearised LMMSE AoA fusion, residual and claimed-vs-fused distance |
| `src/spoof_sim.py` | **simulated** hidden-true-position spoofing |
| `src/features.py` | per (vehicle, 10 s window) features; `sim_*` columns are simulated |
| `src/graph_builder.py`, `src/classifiers/graph_gbdt.py` | spatiotemporal graph + node2vec + XGBoost |
| `src/evaluate.py`, `src/ablation.py` | metrics, curves, per-class tables, ablation |

## Headline results (window level; 17,888 test windows, split by sender)
| model | F1 | ROC-AUC | FPR | simulated? |
|---|---|---|---|---|
| features only | 0.969 | 0.979 | 0.0034 | no |
| graph (node2vec) | 0.970 | 0.978 | 0.0018 | no |
| fusion | 0.988 | 0.9985 | 0.0022 | **YES** |
| graph + fusion | 0.987 | 0.9983 | 0.0020 | **YES** |

Ablation (SIMULATED, mean of 5 splits): dropping `claim_fused_dist` costs 0.026 F1
(0.9850 to 0.9588, back to the no-fusion level of 0.9589); dropping the residual
`angular_consistency_score` costs nothing. See the report for caveats.

## Observer coverage note (Phase 3)
The map is about 1.5 x 1.5 km. The guide's defaults (3 UAVs, 250 m) leave 39% of messages
unseen and none seen by 2+ UAVs. Share seen by 2+ observers (k-means placement):
3 UAVs/250 m 0% | 3/500 m 27% | 6/250 m 13% | 10/250 m 37% | **15/250 m 72% (used)** | 10/350 m 74%.

## Live demo app (`app/`)
A dark-themed FastAPI dashboard with six tabs: Overview (animated headline numbers), Live detector (probability gauge, probability, speed and feature-importance charts), Drone simulator (drag a car and a claimed position; drones' bearings are fused live in the browser to show how a lie is exposed - an illustration, not a result), Results (comparison, ablation, per-class and spoof-detection charts, simulated results hatched and badged), Pipeline (architecture diagram) and Limits. The live detector scores one vehicle's message trace with the **measured Phase 2 baseline** (compact Random Forest, F1 0.978 / ROC-AUC 0.981 on held-out senders; 0.8 MB). It does **not** use the simulated fusion features: those need several UAV observers, which a single trace cannot provide. Pick one of 33 held-out sample vehicles or paste your own CSV (`timestamp,x,y,speed,heading[,acl,pos_noise,spd_noise]`, one vehicle, 10+ messages). The page warns when noise columns are missing or the trace is short, because both bias the verdict towards "attacker".

Run locally (after the install step above):
```bash
venv/Scripts/python -m pip install fastapi uvicorn httpx
venv/Scripts/python -m uvicorn app.server:app --port 8000      # open http://localhost:8000
```
Rebuild the model, samples and results data (needs `data/processed/merged.parquet` and `results/metrics/`): `venv/Scripts/python -m app.build_assets`.

Deploy: `Dockerfile` + `requirements-app.txt` (pinned versions; the model file must be loaded with the same scikit-learn, 1.7.2) package only the demo, no dataset needed. Any container host works (Render, Fly.io, Railway, Cloud Run); the app reads the `PORT` environment variable.
```bash
docker build -t veremi-demo .
docker run -p 8000:8000 veremi-demo
```
The Dockerfile has not been built or tested (Docker is not installed on the development machine); the app itself is covered by `tests/test_app.py`. Nothing has been published anywhere.

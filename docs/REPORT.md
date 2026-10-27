# Cooperative multi-UAV misbehaviour detection on VeReMi Extension: final report

**Status of the evidence.** Results are split into two kinds and every table says which:
- **Measured** - computed on the real VeReMi Extension table (baseline, kinematic features, graph embeddings).
- **SIMULATED** - anything using the cooperative AoA-fusion features (`sim_*`). The dataset has no ground-truth transmitter positions, so bearings for attackers come from a synthetic hidden true position (claimed minus a 30-300 m random offset). These numbers show what the method could do under that assumption; they are **not** measured VeReMi performance.

## 1. Data
Mendeley `mixalldata_clean.csv`: 3,194,808 messages, 24,663 senders, 40.5% attack messages over 19 numeric attack classes (names not mapped), one row per message with claimed position, speed, acceleration, heading and noise columns. It has **no receiver column and no true positions**, which differs from the implementation guide's assumption of raw per-vehicle logs. Experiments use 20% of senders (whole senders, so message sequences stay intact), about 625k messages, aggregated to 59,108 (vehicle, 10 s) windows, 30.2% attacker windows. Train/test splits are by sender, so no vehicle appears in both.

## 2. Methods
1. **Baseline (Phase 2, measured, message level).** Random Forest on proxies built from each sender's own past messages: reported noise, claimed-vs-implied speed gap, heading change, inter-message gap deviation, and a 0.7/0.3 blend of direct and pseudonym-level trust (the weights and the 5 m/s plausibility threshold are my choices). Held-out F1 0.978 (attacker class), ROC-AUC 0.982. This is stronger than a true single-observer baseline because it uses sender history.
2. **Virtual UAV overlay (Phase 3).** 15 observers at 100 m altitude, k-means placed over traffic, 250 m coverage, ideal azimuth/elevation bearings. The guide's 3 UAVs left no overlap (0% of messages seen by 2+); 15 UAVs gives 72%.
3. **Cooperative AoA fusion (Phase 4).** Sequential linearised LMMSE (EKF-style, 2 passes) over azimuths with 2 degrees of simulated Gaussian bearing error, per (vehicle, 1 s). Outputs the fused position, the per-observer residual score (defined only with 3+ observers, since 2 bearings fit exactly) and `claim_fused_dist`, the distance between claimed and fused position. Verified on hand-built geometry and on rows with variable observer counts (a NaN-masking bug was found and fixed this way).
4. **Spoofing simulation (Phase 4b/8b, SIMULATED).** Attackers' bearings come from a hidden true position. Which UAVs hear a message is decided from the claimed position, because using the true position made coverage counts leak the label (found in the Phase 8 ablation, fixed in Phase 8b).
5. **Features (Phase 5).** Per (vehicle, 10 s): window-mean baseline features, speed variance, heading-change rate, position variance, and `sim_*` fusion aggregates (median and max claimed-vs-fused distance clipped at 500 m, mean observer count, residual score, single-observer fraction, fused sub-window count). 74.6% of windows have at least one fused sub-window; the rest keep NaN in `sim_*` (XGBoost handles it).
6. **Graph + classifier (Phase 6).** Nodes are (vehicle, window); temporal edges join consecutive windows, spatial edges join each vehicle to its 5 nearest neighbours within 150 m (claimed positions). 32-d node2vec embeddings (no labels used) are concatenated with the features and fed to XGBoost.

## 3. Results (window level, 17,888 test windows, 5,439 attack)
| model | F1 | ROC-AUC | PR-AUC | FPR | evidence |
|---|---|---|---|---|---|
| features only | 0.969 | 0.979 | 0.979 | 0.0034 | measured |
| graph (node2vec) | 0.970 | 0.978 | 0.978 | 0.0018 | measured |
| fusion | 0.988 | 0.9985 | 0.997 | 0.0022 | **SIMULATED** |
| graph + fusion | 0.987 | 0.9983 | 0.997 | 0.0020 | **SIMULATED** |

Source: `results/metrics/comparison.csv`; curves and confusion matrices in `results/figures/` (simulated variants dashed, hatched or tagged).

**Per attack class** (`results/metrics/per_attack_type.csv`, class vs genuine). Most classes reach F1 0.87-0.97 without fusion. Class 2 is invisible without fusion (F1 0.00, AUC 0.51) and class 9 is weak (0.74). Simulated fusion lifts them to 0.79 and 0.87, but only because the simulation spoofs every class.

**Spoof detection alone (SIMULATED, sender-windows with 2+ observers).** ROC-AUC of `claim_fused_dist`: 0.962 for 30-300 m offsets, 0.874 for 10-50 m. The residual score scores 0.395 and 0.455 (below 0.5).

**Ablation (SIMULATED, mean of 5 sender-level splits, F1):**

| variant | no graph | delta | with graph | delta |
|---|---|---|---|---|
| full (all `sim_*`) | 0.9850 | - | 0.9841 | - |
| drop `claim_fused_dist` | 0.9588 | -0.0261 | 0.9579 | -0.0262 |
| drop residual score | 0.9852 | +0.0002 | 0.9838 | -0.0004 |
| drop both (coverage counts left) | 0.9588 | -0.0261 | 0.9577 | -0.0264 |
| no fusion at all (measured) | 0.9589 | -0.0261 | 0.9578 | -0.0263 |

## 4. Findings
1. Strong kinematic and trust features already give F1 about 0.97 on real data; node2vec graph embeddings add essentially nothing (F1 +0.001, lower FPR).
2. **The residual-based angular-consistency score, the guide's central fusion feature, carries no spoofing signal** (ablation delta within noise; AUC below 0.5). All observers measure the same physical transmitter, so their bearings agree whatever position is claimed. The usable cooperative signal is `claim_fused_dist`, the gap between the claimed and bearing-derived position. The residual would matter only for inconsistent bearings (several physical sources), which this simulation does not model.
3. Under the simulation, `claim_fused_dist` is worth about +0.026 F1 and recovers most of the two hardest classes; this is a property of the assumption.
4. Two bugs were found only because of tests and the ablation: padded-NaN poisoning of the fusion update, and coverage counts leaking the label (which had inflated the ablation by about 0.013 F1).

## 5. Limitations
- **Fusion results are simulated and optimistic**: every attack class is treated as position spoofing (30-300 m); real VeReMi attacks include speed and DoS types that do not falsify position. Real AoA hardware error, UAV positioning error and GPS error beyond the dataset's noise columns are not modelled; coverage from the claimed position is slightly unphysical.
- No true transmitter positions exist in this table, so the cooperative layer cannot be validated on real data here. Validating it needs the raw VeReMi Extension traces or a simulator that exports true positions.
- The task is attack-vs-genuine over 19 classes, not Sybil-only (class ids not mapped to names).
- The baseline uses sender history in place of receiver-side signal data (none in the table); trust weights are my choices.
- 20% sender sample; one fixed split seed for the Phase 7 tables (5 seeds for the ablation); no hyperparameter tuning; node2vec settings untuned.
- Phase 9 (collusion scenario, optional in the plan) was not attempted.

## 6. Reproduction
See the README. All numbers above come from `src.evaluate`, `src.ablation`, `src.spoof_sim` and `src/train.py --variant baseline`; CSVs are in `results/metrics/`.

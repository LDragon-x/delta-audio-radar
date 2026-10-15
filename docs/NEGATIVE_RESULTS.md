# Negative Results & Overturned Routes

This project keeps its falsified routes on record, each with the quantitative
evidence that killed it. Four ML approaches were tried and overturned before the
current pipeline (hotkey masking → voiceprint → frame-level Wiener-gain model)
stabilized. Dates are from the engineering journal.

## 1. Audio fingerprinting for own/enemy separation — overturned by field data (09-27)

Synthetic acceptance passed cleanly (own d=0.000 vs enemy d=0.124 on a
synthesized scene). Field calibration then showed real enemy footsteps sitting at
d≈0.12 from the library — the synthetic and real distributions simply do not
overlap. The 12-band log-spectrum + cosine-distance fingerprint stayed in the
pipeline as a weak prior, no longer trusted as a separator.

Lesson: synthetic acceptance data cannot calibrate thresholds for real-world
audio; field data is the only referee.

## 2. PCEN (per-channel energy normalization) features — uniformly negative (09-30)

Implemented per Zhao et al. 2024 (+12 F1 reported on their data). On our
baking-race protocol (same labeled samples, per-session hold-out, 5-NN):
log-mel all = 39%/88% (task A/B) vs PCEN all = 41%/73%, and every single-feature
PCEN variant scored worse still. Implementation verified correct (background
zeroed, single-pulse response, dynamic compression) — the effect is genuinely
system-dependent. The literature number did not transfer.

## 3. Template subtraction (robotics ego-noise classic) — no signal to subtract (10-01)

Following the ego-noise literature (pre-recorded action templates subtracted at
runtime by action timestamps), built a baseline using our hotkey timeline as the
"action" reference. Result: correlation of own segments with the template 0.50
vs 0.49 for enemy segments — indistinguishable. Root cause: positional reverb
makes the spectrum of "the same" footstep drift per instance, unlike a robot
motor. Robotics' oldest trick does not transfer to reverberant game audio.

## 4. Footstep-cadence analysis (bot vs human) — AUC ≈ 0.5 (10-01)

Hypothesis: bots are metronomes (fixed inter-step intervals), humans vary pace.
Built `tool/cadence.py` with six regularity metrics (step rate, beat jitter,
±20% hit ratio, stereo drift, ...). Every metric landed at AUC ≈ 0.5 — and beat
jitter ran *opposite* to the hypothesis (bots jittered more), because segments
mixed multiple walkers and detection timing is noisy. Only 12 ground-truth
segments existed at the time. Module kept for re-evaluation as labeled data grows.

## Control experiment: DeepFilterNet3 (10-01)

A general-purpose open-source speech-enhancement model, run as an out-of-the-box
baseline on the same metric protocol: **96.9% false suppression** of enemy
footsteps. A task-blind generic model cannot do selective ego-noise suppression —
empirical justification for a task-specific pipeline.

## What survived

- Hold-out evaluation × 3 seeds, median reported (never the best run)
- Per-session hold-out (no same-match leakage between train and test)
- Every threshold documented with its scan range and chosen operating point

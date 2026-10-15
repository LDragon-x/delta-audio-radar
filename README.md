# Delta Audio Radar

Real-time game-audio event perception and **ego-noise suppression** on the Windows
loopback path — no injection, no memory reading, no kernel drivers.

The system listens to whatever the sound card plays (WASAPI loopback), detects
sound events (footsteps / gunfire / glass / actions), decides which of them are
*your own* sounds, and can suppress them in real time before they reach your
headphones — a "double extraction" cascade: first remove yourself from the audio,
then let perception run on the clean stream.

Built in a **human-led, AI-assisted workflow**: the owner defined requirements,
captured and annotated all field data, and accepted every change; implementation
was executed in pair-programming with an AI coding agent. All quantitative claims
below come from recorded experiments; every falsified route is documented in
[`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) instead of being deleted.

## Architecture

```
Game audio → WASAPI loopback capture (pyaudiowpatch callback)
  → Event detection (energy envelope / band features)
  → Ego-noise suppression, three defense lines:
       1. Global hotkey masking (WH_KEYBOARD_LL hook: WASD held = own footsteps,
          mouse = own gunfire, X = skill window, movement latch for tap-strafers)
       2. Voiceprint fingerprint library (12-band log-spectrum + cosine distance)
       3. Loudness-centering rule
     plus ML: frame-level GRU with Wiener-gain supervision (Mix-and-Mask training),
     posture-conditional gain shaping (idle g² / moving g³)
  → Real-time chain (virtual-sound-card closed loop; F9 bypass, F10 quit)
  → Output layer (self-tagged events kept on disk as training negative samples)

Data flywheel: capture tool (audio + screen + keyboard edges + mouse trajectory)
  → annotation UIs (segment / timeline-range, smart episode packing)
  → fingerprint library / models → engine feedback
```

## Quickstart

```bash
pip install -r requirements.txt

# Offline acceptance test — synthetic signals, no audio device needed.
# Runs the DSP chain end-to-end and checks 14 assertions (detection, masking,
# voiceprint self/enemy separation):
python main.py --demo
# → ends with "DEMO PASS"

# List loopback capture devices:
python main.py --list

# Live debug panel on your system output:
python main.py --loopback
```

Field capture, annotation and training tools live in `tool/`:

| Script | Purpose |
|---|---|
| `tool/collector.py` | Capture a session: loopback audio + screen + keyboard edges + mouse trajectory (global hotkeys, F10 stop) |
| `tool/selfvoice.py` | Build / match / manage the own-voiceprint fingerprint library |
| `tool/labeling_ui.py` | Browser annotation UI (segment and timeline-range modes) |
| `tool/smart_pack.py` | Auto-select high-value clips for annotation (engagement-window priority) |
| `tool/feat_eval.py` | Feature baking race: N feature sets × tasks, per-session hold-out |
| `tool/learn_embed.py` | Embedding classifier + transductive adaptation training/eval |
| `tool/train_frame.py` | Frame-level model training (Mix-and-Mask, Wiener-gain targets) |
| `tool/rt_suppress.py` | Real-time suppression chain (virtual sound card closed loop) |
| `tool/cadence.py` | Footstep-cadence analysis (kept after a negative result) |

## Measured results (from recorded experiments)

- Own-footstep suppression: deep-suppression rate 53–57% at −23.5~−24.5 dB median,
  false suppression of other sounds 17.4% (threshold scan τ=0.35)
- Full-chain latency 0.37 s (0.65 s → 0.37 s via truncation-augmented training:
  half-window inference 71.7% vs full-window 65.2%)
- Acceptance: 14-assertion `--demo` regression, green on every change

See `docs/NEGATIVE_RESULTS.md` for the four overturned routes (audio fingerprinting,
PCEN, template subtraction, cadence analysis) and the DeepFilterNet3 baseline —
each with its quantitative evidence.

## Data & models

Private field data (recorded sessions, annotated clips, voiceprint libraries,
trained weights) is **not** in this repository — it contains personal voice data
and copyrighted game audio. Everything is regenerable from the tools above;
`python main.py --demo` verifies the full DSP chain on synthetic signals alone.

## Red lines

No process injection, no screen reading of the game, no kernel drivers, no
automation of in-game input. The project observes its own audio output only.
Please respect your game's terms of service.

## Layout

```
main.py            engine entry + offline demo acceptance
dsp/               detection, features (log-mel), fingerprints, frame model
capture/           loopback/file/synthetic audio sources, keyboard & screen capture
ui/                live debug panel
state/             run-state tracker
tool/              capture, annotation, training, evaluation, real-time chain
docs/              design decisions, roadmap, UI spec, negative results
tests/             pytest suite (signal properties + demo acceptance)
```

## License

MIT (see `LICENSE`). Game data referenced by configuration is property of its
respective owners; none of it is distributed here.

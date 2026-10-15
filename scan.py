"""快速扫描整条音频：每 30 秒窗口统计触发数，给录像画"声音结构地图"。"""
import json
import sys

sys.path.insert(0, ".")
import numpy as np

cfg = json.load(open("config.json", encoding="utf-8"))
import wave

from dsp.pipeline import Engine

w = wave.open("testdata/audio.wav", "rb")
sr = w.getframerate()
raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
raw = raw.reshape(-1, w.getnchannels())[:, :2]
w.close()

engine = Engine(cfg)
block = cfg["block_size"]
WIN = 30.0
counts = {}
tiers = {}
flats = []
for i in range(0, len(raw) - block, block):
    t = i / sr
    snap = engine.process(t, raw[i:i + block])
    win = int(t // WIN)
    ev = snap.get("event")
    if ev and ev["type"] in ("footstep", "gunshot"):
        counts[win] = counts.get(win, 0) + 1
        if ev["type"] == "footstep":
            tiers[win] = tiers.get(win, 0) + (1 if snap["tier"] == "近" else 0)
            flats.append(snap["feats"]["low_flat"])

print("时间窗    事件数  其中近档脚步")
for win in sorted(counts):
    print(f"{win*WIN:6.0f}s  {counts[win]:4d}    {tiers.get(win, 0):4d}")
if flats:
    print(f"\n全片脚步平坦度: min={min(flats):.3f} med={sorted(flats)[len(flats)//2]:.3f} "
          f"p90={sorted(flats)[int(len(flats)*0.9)]:.3f} max={max(flats):.3f} n={len(flats)}")

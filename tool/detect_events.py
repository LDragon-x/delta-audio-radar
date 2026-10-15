# -*- coding: utf-8 -*-
"""离线事件检测：对已有 session.wav 跑 dsp.pipeline.Engine，补 auto_events.jsonl。

动机（2026-10-01）：quick_rec.py 只录音频+键鼠，不跑引擎；而 pose_label /
train_pose_protos 需要 auto_events.jsonl 的脚步事件。本脚本按 collector 同款
喂法（48k 立体声块）离线补检。亦可用于任何缺 auto_events 的老会话。

用法：python tool/detect_events.py <会话目录名> [更多会话...]
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dsp.pipeline import Engine  # noqa: E402


def main():
    cfg = json.load(open(ROOT / "config.json", encoding="utf-8"))
    for sid in sys.argv[1:]:
        d = ROOT / "sessions" / sid
        wv = d / "session.wav"
        if not wv.exists():
            print(f"{sid}: 无 session.wav，跳过")
            continue
        w = wave.open(str(wv), "rb")
        sr, ch = w.getframerate(), w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        w.close()
        audio = raw.reshape(-1, ch).astype(np.float32) / 32768.0
        if sr != cfg["sample_rate"]:
            print(f"{sid}: 采样率 {sr} ≠ {cfg['sample_rate']}，跳过")
            continue
        eng = Engine(cfg)
        block = 2048
        evs = []
        for i in range(0, len(audio) - block + 1, block):
            t = i / sr
            frame = audio[i:i + block]
            snap = eng.process(t, frame)
            ev = snap.get("event")
            if ev and ev["type"] in ("footstep", "gunshot", "bot_footstep",
                                      "glass", "action"):
                f = snap["feats"]
                evs.append({"t": round(ev["t"], 3), "type": ev["type"],
                            "pan": round(ev["pan"], 3),
                            "self": bool(ev.get("self")),
                            "self_reason": ev.get("self_reason", ""),
                            "feat": {k: round(float(f[k]), 6) for k in
                                     ("low", "mid", "high", "low_ratio",
                                      "high_ratio", "corr", "low_flat", "rms")}})
        out = d / "auto_events.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for e in evs:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        steps = sum(1 for e in evs if e["type"] == "footstep")
        print(f"{sid}: {len(evs)} 事件（脚步 {steps}）→ {out.name}")


if __name__ == "__main__":
    main()

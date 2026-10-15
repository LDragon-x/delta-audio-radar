# -*- coding: utf-8 -*-
"""极简采集器 v2（应急版）：录 VB 虚拟线输入 + 按键流，给素材录制用。

v1（loopback 直采 GAMING）废弃原因（2026-10-01 实锤）：
  soundcard 的 WASAPI loopback 在本机音频会话频繁切换后内部崩死
  （IAudioCaptureClient 空指针，record() 静默返回零/抛异常），
  独立复现同参数也复现——该路径不可托付。
v2 改录 "入梦麦克风 (VB-Audio Cable A)"（普通录音设备，rt_suppress 同款源），
  拓扑：游戏→VB线→[本采集器录原始音] + [rt_suppress 消噪→GAMING 给用户听]，
  录到的是未处理原声，用户听到的仍是消噪后声音，互不耽误。
用法：python tool/quick_rec.py [N秒自动停]
"""
import json
import math
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capture.keys import KeyLogger  # noqa: E402

CABLE_KEYWORD = "VB-Audio Cable A"


def find_cable():
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and CABLE_KEYWORD in d["name"]:
            return i, d["name"]
    raise RuntimeError(f"找不到 {CABLE_KEYWORD} 输入设备")


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else None
    import sounddevice as sd
    d = ROOT / "sessions" / time.strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    wav = wave.open(str(d / "session.wav"), "wb")
    wav.setnchannels(2)
    wav.setsampwidth(2)
    wav.setframerate(48000)
    dev, name = find_cable()
    t0 = time.perf_counter()
    keys = KeyLogger(d / "keys.jsonl", t0_fn=lambda: t0,
                     keys=["w", "a", "s", "d", "shift", "c", "z", "space",
                           "x", "r", "1", "2", "3", "q", "b"],
                     mouse_path=d / "mouse.jsonl")
    keys.start()
    stream = sd.InputStream(device=dev, channels=2, samplerate=48000,
                            blocksize=2048, dtype="float32")
    stream.start()
    print(f"quick_rec v2: {name} (#{dev}) → {d}", flush=True)
    n_bytes = 0
    try:
        while True:
            frame, _ = stream.read(2048)
            pcm = (np.clip(frame, -1, 1) * 32767).astype("<i2")
            wav.writeframes(pcm.tobytes())
            n_bytes += pcm.nbytes
            t = time.perf_counter() - t0
            if n_bytes % (48000 * 2 * 2 * 30) < pcm.nbytes:
                rms = float((frame ** 2).mean() ** 0.5)
                print(f"  t={t:7.1f}s rms={20 * math.log10(rms + 1e-9):6.1f}dB",
                      flush=True)
            if seconds and t >= seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        stream.stop()
        stream.close()
        wav.close()
        print(f"saved {d}", flush=True)


if __name__ == "__main__":
    main()

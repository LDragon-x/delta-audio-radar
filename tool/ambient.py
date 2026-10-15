"""环境音档案提取：把一段游戏音频的"安静时刻"凝成声纹档案。

用途：后续做环境音屏蔽（频谱减法/白化）时的参照底噪。
每个地图/场景存一份档案：低活跃帧的频段能量均值 + 质感特征。

用法：
  python tool/ambient.py --file testdata/az3_clean.wav --tag 航天基地
  python tool/ambient.py --list          # 查看已有档案
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp.features import extract_features
from dsp.pipeline import Engine

OUT_DIR = Path("samples/ambient")


def extract_profile(cfg, path, tag):
    w = wave.open(path, "rb")
    sr = w.getframerate()
    raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    data = raw.reshape(-1, w.getnchannels())[:, :2]
    w.close()

    engine = Engine(cfg)
    block = cfg["block_size"]
    quiet = []          # (feats) 无事件且能量低于中位数的帧
    fulls = []
    pending = []
    for i in range(0, len(data) - block, block):
        t = i / sr
        snap = engine.process(t, data[i:i + block])
        f = snap["feats"]
        ev = snap.get("event")
        # 事件帧及其后 0.7s 都排除（枪声尾巴会污染底噪统计）
        if ev:
            pending.append((t, ev["type"]))
        recent_event = any(t - et < 0.7 for et, _ in pending)
        if not recent_event:
            fulls.append(f["full"])
            quiet.append(f)

    if not quiet:
        raise RuntimeError("没有找到足够的安静帧")
    med = float(np.median(fulls))
    calm = [f for f in quiet if f["full"] < med * 0.8] or quiet

    def avg(key):
        return float(np.mean([f[key] for f in calm]))

    profile = {
        "tag": tag,
        "source": str(path),
        "duration_s": round(len(data) / sr, 1),
        "calm_frames": len(calm),
        "bands": {k: avg(k) for k in ("low", "mid", "high", "full")},
        "low_ratio": avg("low_ratio"),
        "high_ratio": avg("high_ratio"),
        "low_flat": avg("low_flat"),
        "corr": avg("corr"),
        "usage": "环境音屏蔽参照：以 bands 为底噪曲线做频谱减法（v0.2 DSP 接入）",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tag}.json"
    out.write_text(json.dumps(profile, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已存 {out}（安静帧 {len(calm)}，{profile['duration_s']}s 音频）")
    print(f"  频段 low={profile['bands']['low']:.2e} mid={profile['bands']['mid']:.2e} "
          f"high={profile['bands']['high']:.2e} corr={profile['corr']:.2f}")
    return profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--tag", help="档案名（建议用地图/场景名）")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for p in sorted(OUT_DIR.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            print(f"{p.stem:<12} {d['duration_s']:7.1f}s  安静帧{d['calm_frames']:<5} "
                  f"low={d['bands']['low']:.2e} corr={d['corr']:.2f}")
        return

    cfg = json.load(open("config.json", encoding="utf-8"))
    extract_profile(cfg, args.file, args.tag)


if __name__ == "__main__":
    main()

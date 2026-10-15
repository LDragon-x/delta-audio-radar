"""log-mel 纹理版学习管线 v2：直接从标注 clips 的音频抽精细指纹训练。

v1（learn.py）用引擎块级频带特征（8 维），实测分不开真人/人机脚步。
v2 思路（用户点子）：人机走不出真人的步伐——换 64 频带 log-mel 纹理
指纹 + 节奏上下文，看小样本下能不能分开。样本直接从 labeling/clips
的 wav 抽（标注过的片段就是金标准音频，不用时间对齐）。

用法：
  python tool/learn_mel.py            # 留一法验证 + 训练存 samples/model/knn_mel.json
  python tool/learn_mel.py --stats    # 只看数据集统计
"""
import argparse
import csv
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_map import class_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "sessions"
MODEL_PATH = ROOT / "samples" / "model" / "knn_mel.json"

CLASSES = ["footstep", "bot_footstep", "gunshot", "self_footstep", "self_gunshot",
           "broadcast", "loot", "bot_vocal", "melee", "skill", "rescue", "other"]
WIN = 0.35       # 事件核窗（秒）：取 clip 中部峰附近
N_FFT = 1024
HOP = 256
N_MELS = 64
FMIN, FMAX = 60, 10000


def melbank(sr, n_fft, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    pts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(pts) / sr).astype(int)
    top = n_fft // 2
    fb = np.zeros((n_mels, top + 1))
    for m in range(n_mels):
        a, b, c = bins[m], bins[m + 1], bins[m + 2]
        b = max(b, a + 1)
        c = max(c, b + 1)
        b, c = min(b, top), min(c, top)
        if b <= a or c <= b:
            continue
        fb[m, a:b] = np.linspace(0, 1, b - a, endpoint=False)
        fb[m, b:c] = np.linspace(1, 0, c - b, endpoint=False)
    return fb


def logmel_vec(path):
    """clip wav → 64 频带 log-mel 纹理向量（固定维度，峰对齐 0.35s 窗）。"""
    w = wave.open(str(path), "rb")
    sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
    raw = w.readframes(n)
    w.close()
    if sr != 48000 or n < 2048:
        return None
    x = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch).astype(np.float32) / 32768
    x = x.mean(axis=1)
    n0 = int(WIN * sr)
    # 峰对齐：±0.5s 内能量峰（与 virtual_scene.cut_kernel 同思路）
    c = len(x) // 2
    win = x[max(0, c - sr // 2): c + sr // 2]
    peak = int(np.argmax(np.convolve(win ** 2, np.ones(256) / 256, "same")))
    p = max(0, c - sr // 2) + peak
    seg = x[max(0, p - n0 // 2): p + n0 // 2]
    if len(seg) < n0:
        seg = np.pad(seg, (0, n0 - len(seg)))
    frames = np.stack([seg[i:i + N_FFT] * np.hanning(N_FFT)
                       for i in range(0, len(seg) - N_FFT + 1, HOP)])
    powspec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    logm = np.log(powspec @ melbank(sr, N_FFT).T + 1e-10)
    v = logm.flatten()
    return (v - v.mean()) / (v.std() + 1e-9)


def load_dataset():
    """[(vec, cls, clip)]：所有会话 labeling clips × 用户标注（class_of 归一）。"""
    samples, stats = [], {}
    for d in sorted(SESSIONS.glob("*")):
        ldir = d / "labeling"
        uf, cdir = ldir / "user_labels.csv", ldir / "clips"
        if not (uf.exists() and cdir.exists()):
            continue
        seen = set()
        with open(uf, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if not r.get("label"):
                    continue
                wav = cdir / r["clip"]
                if not wav.exists():
                    continue
                for lab in r["label"].split(";"):
                    cls = class_of(lab.strip())
                    if cls not in CLASSES or (r["clip"], cls) in seen:
                        continue
                    seen.add((r["clip"], cls))
                    v = logmel_vec(wav)
                    if v is None:
                        continue
                    samples.append((v, cls, f"{d.name}/{r['clip']}"))
                    stats[cls] = stats.get(cls, 0) + 1
    return samples, stats


def loo_eval(samples, k=3):
    """留一法 k-NN（距离加权）：小样本最诚实的验证方式。"""
    X = np.array([s[0] for s in samples])
    y = [s[1] for s in samples]
    n = len(samples)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    preds = []
    for i in range(n):
        kk = np.argsort(D[i])[:k]
        votes = {}
        for j in kk:
            votes[y[j]] = votes.get(y[j], 0) + 1.0 / (D[i, j] + 1e-9)
        preds.append(max(votes, key=votes.get))
    conf = {}
    for cls in sorted(set(y)):
        tp = sum(1 for i in range(n) if y[i] == cls and preds[i] == cls)
        fp = sum(1 for i in range(n) if y[i] != cls and preds[i] == cls)
        fn = sum(1 for i in range(n) if y[i] == cls and preds[i] != cls)
        conf[cls] = {"precision": round(tp / max(1, tp + fp), 2),
                     "recall": round(tp / max(1, tp + fn), 2),
                     "support": sum(1 for yy in y if yy == cls)}
    return conf, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    samples, stats = load_dataset()
    print(f"数据集：{stats}，共 {len(samples)} 条")
    if args.stats or not samples:
        if not samples:
            print("没有标注 clips——先去边听边标标几段")
        return

    conf, _ = loo_eval(samples)
    print("\n留一法验证（k=3，距离加权）：")
    for cls, r in sorted(conf.items()):
        print(f"  {cls:<14} 精确率 {r['precision']:.2f}  召回率 {r['recall']:.2f}  "
              f"样本 {r['support']}")

    # 全量训练存模型（供后续引擎接线/回放评估）
    X = np.array([s[0] for s in samples])
    y = [s[1] for s in samples]
    model = {"k": 3, "win": WIN, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS,
             "fmin": FMIN, "fmax": FMAX, "X": X.tolist(), "y": y,
             "training_stats": stats}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False))
    print(f"\n模型已存 {MODEL_PATH}")


if __name__ == "__main__":
    main()

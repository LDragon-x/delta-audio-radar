"""录屏-音频时间轴重建（DTW）：救编码跟不上导致时间轴压缩的旧会话。

原理：视频逐帧画面变化（枪战/跑动→大，加载/静止→小）和音频能量包络
（枪声/脚步→大，静音→小）是同一场景的两个观测，按单调映射做动态时间
规整，反推每帧的真实会话时刻，写 video_frames.json 供标注页精确对位。

用法：
  python tool/resync_video.py --session sessions/<时间戳>

只写 video_frames.json，不动原 mp4；已有帧表的会话默认跳过（--force 重算）。
"""
import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIDFPS = 15          # 写盘时用的名义帧率（mp4 时间轴 = 帧号/15）
DMIN, DMAX = 0, 6    # 每帧对应的音频 bin 步进范围（0~0.4s/帧）
SLOPE_PRIOR = 0.05   # 全局斜率弱先验权重（低信号区防漂）


def audio_env(wav_path, bin_hz=VIDFPS):
    """音频能量包络，按 1/VIDFPS 秒分箱，log+zscore。"""
    w = wave.open(str(wav_path), "rb")
    sr, n = w.getframerate(), w.getnframes()
    ch = w.getnchannels()
    raw = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)
    w.close()
    raw = raw.reshape(-1, ch).mean(axis=1)
    nb = int(np.ceil(len(raw) / sr * bin_hz))
    env = np.zeros(nb, dtype=np.float32)
    idx = (np.arange(len(raw)) * bin_hz / sr).astype(np.int64)
    np.add.at(env, np.clip(idx, 0, nb - 1), np.abs(raw))
    cnt = np.zeros(nb, dtype=np.float32)
    np.add.at(cnt, np.clip(idx, 0, nb - 1), 1.0)
    env /= np.maximum(cnt, 1)
    return np.log(env + 1e-4), sr


def video_motion(mp4_path):
    """逐帧画面变化量：灰度缩到 64x36 后的帧间平均绝对差，log+zscore。"""
    import imageio.v2 as imageio
    from PIL import Image
    r = imageio.get_reader(str(mp4_path))
    prev = None
    motion = []
    for fr in r:
        g = np.asarray(Image.fromarray(fr).resize((64, 36), Image.BILINEAR)
                       .convert("L"), dtype=np.float32)
        if prev is not None:
            motion.append(np.abs(g - prev).mean())
        prev = g
    r.close()
    return np.log(np.asarray(motion, dtype=np.float32) + 1e-4)


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def dtw_map(za, zd):
    """za: 音频包络 [J]，zd: 帧变化 [N]。返回每帧对应的音频 bin j(i)。

    DP：C[i,j] = (za[j]-zd[i])^2 + λ(j-i*slope_g)² + min_k C[i-1,j-k]，
    k∈[DMIN,DMAX]。窗口限制在全局斜率 ±W 内，内存只留 argmin 矩阵。
    """
    N, J = len(zd), len(za)
    slope_g = J / N
    W = int(0.30 * slope_g * N / N * 1)  # 每侧富余窗口（按全局斜率的30%）
    W = max(200, int(0.35 * slope_g))
    # 窗口内列坐标：j_lo(i) <= j < j_lo(i)+2W
    j_lo = np.clip((np.arange(N) * slope_g - W).astype(int), 0, J - 1)
    width = min(2 * W + 8, J)
    C = np.full((N, width), np.inf, dtype=np.float32)
    back = np.zeros((N, width), dtype=np.int16)
    # i=0
    js = np.arange(j_lo[0], min(j_lo[0] + width, J))
    C[0, :len(js)] = (za[js] - zd[0]) ** 2 + \
        SLOPE_PRIOR * (js - 0 * slope_g) ** 2
    for i in range(1, N):
        lo = j_lo[i]
        js = np.arange(lo, min(lo + width, J))
        cost = (za[js] - zd[i]) ** 2 + \
            SLOPE_PRIOR * (js - i * slope_g) ** 2
        best = np.full(len(js), np.inf, dtype=np.float32)
        bestk = np.zeros(len(js), dtype=np.int16)
        prev_lo = j_lo[i - 1]
        for k in range(DMIN, DMAX + 1):
            # C[i-1, j-k] 在上一行窗口内的位置
            pos = (js - k) - prev_lo
            valid = (pos >= 0) & (pos < width)
            cand = np.full(len(js), np.inf, dtype=np.float32)
            cand[valid] = C[i - 1, pos[valid]]
            upd = cand < best
            best[upd] = cand[upd]
            bestk[upd] = k
        C[i, :len(js)] = cost + best
        back[i, :len(js)] = bestk
    # 回溯：终点取最后一行窗口内最小值
    i = N - 1
    wlen = min(width, J - j_lo[i])
    j = j_lo[i] + int(np.argmin(C[i, :wlen]))
    path = np.zeros(N, dtype=np.int64)
    path[N - 1] = j
    for i in range(N - 1, 0, -1):
        k = int(back[i, j - j_lo[i]])
        j -= k
        path[i - 1] = j
    return path  # 每帧 → 音频 bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    d = Path(args.session)
    mp4, wav = d / "video.mp4", d / "session.wav"
    out = d / "video_frames.json"
    if not mp4.exists() or not wav.exists():
        raise SystemExit("缺 video.mp4 或 session.wav")
    if out.exists() and not args.force:
        print(f"已有 {out.name}，跳过（--force 重算）")
        return

    print("1/4 音频包络…")
    env, _ = audio_env(wav)
    za = zscore(env)
    print(f"    {len(za)} bins（{len(za) / VIDFPS:.0f}s）")
    print("2/4 视频逐帧变化…")
    mot = video_motion(mp4)
    zd = zscore(mot)
    print(f"    {len(zd)} 帧")

    print("3/4 DTW 对齐（单调映射）…")
    path = dtw_map(za, zd)
    times = (path / VIDFPS).round(3).tolist()
    assert all(a <= b for a, b in zip(times, times[1:])), "映射非单调"

    # 对齐质量：映射点两侧信号相关
    j = np.clip(path, 1, len(za) - 2)
    corr = float(np.corrcoef(za[j], zd)[0, 1])

    print("4/4 校验…")
    dt = np.diff(times)
    print(f"    帧间隔: 中位 {np.median(dt) * 1000:.0f}ms "
          f"p95 {np.percentile(dt, 95) * 1000:.0f}ms 最大 {dt.max() * 1000:.0f}ms")
    print(f"    映射范围: {times[0]:.1f}s → {times[-1]:.1f}s"
          f"（音频全长 {len(za) / VIDFPS:.0f}s）")
    print(f"    映射点信号相关 r={corr:.2f}")

    # 按键交叉验证：W 按住期间画面变化应高于全局中位
    kf = d / "keys.jsonl"
    if kf.exists():
        holds = []
        down = {}
        for line in kf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e["down"]:
                down[e["key"]] = e["t"]
            elif e["key"] in down:
                holds.append((down.pop(e["key"]), e["t"]))
        med = np.median(mot)
        hit = tot = 0
        for a, b in holds:
            if b - a < 0.3:
                continue
            ia = int(a / (len(mot) / (times[-1] or 1)))  # 粗定位到帧
            ia = int(np.searchsorted(times, a))
            ib = int(np.searchsorted(times, b))
            if 0 <= ia < ib < len(mot):
                tot += 1
                hit += mot[ia:ib].mean() > med
        if tot:
            print(f"    按键验证: {hit}/{tot} 个按住区间画面活动高于中位")

    out.write_text(json.dumps({"fps": VIDFPS, "times": times,
                               "corr": round(corr, 3),
                               "source": "dtw-resync"}))
    print(f"\n已写 {out}——刷新标注页即生效")


if __name__ == "__main__":
    main()

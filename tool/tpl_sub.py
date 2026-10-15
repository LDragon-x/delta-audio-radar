# -*- coding: utf-8 -*-
"""模板减除基线 v0.1（Ince 2009 思路的单通道游戏版）。

思路：自己角色的脚步在游戏内是同一套音源，频谱形状高度稳定——
把已标注的 self:脚步 片段做成「模板谱」（中位数，抗离群），对目标
片段估计增益 g 后直接幅度域减除：

    resid_mag = max(mag - alpha * g * T, beta * mag)

alpha 过减除系数，beta 谱地板（防音乐性噪声）。相位用原始相位 ISTFT，
直接能听出清效果。

诚实口径：模板来源 = 除留出局外的所有历史局；留出局（默认最新一局）
只做验证。报两个数：
  - 自己段抑制 dB（越低越好，深度负值=自己脚步被压掉）
  - 敌人/人机段误伤 dB（越接近 0 越好，0=没误伤别人）

用法：
  python tool/tpl_sub.py              # 留出最新一局，alpha 扫描
  python tool/tpl_sub.py --alpha 1.5  # 指定过减除系数出 demo 音频
"""
import argparse
import csv
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SR = 16000
N_FFT = 512
HOP = 128
BETA = 0.10          # 谱地板：残留不低于原幅度的 10%
SELF_TAG = "self:脚步"
DEMO_N_SELF = 4
DEMO_N_FOE = 2


CLIP_SAMPLES = 38400   # 统一窗长 2.4s@16k，中心裁剪/补零对齐模板


def fix_len(x, n=CLIP_SAMPLES):
    """中心裁剪/补零到定长，保证所有片段频谱同形可对齐。"""
    if len(x) >= n:
        s = (len(x) - n) // 2
        return x[s:s + n]
    return np.pad(x, (0, n - len(x)))


def load_clip_wav(path):
    """clips wav (48k 立体声) → 16k 单声道，与 feat_eval 同口径抽稀。"""
    with wave.open(str(path), "rb") as w:
        sr, ch, sw, n = (w.getframerate(), w.getnchannels(),
                         w.getsampwidth(), w.getnframes())
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch).mean(axis=1)
    return a[::(sr // SR)].astype(np.float32) / 32768.0


def stft_mag(x):
    win = np.hanning(N_FFT).astype(np.float32)
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * win
    spec = np.fft.rfft(frames, axis=1)
    return np.abs(spec).astype(np.float32), np.angle(spec).astype(np.float32)


def istft_mag(mag, phase):
    frames = np.fft.irfft(mag * np.exp(1j * phase), n=N_FFT, axis=1)
    win = np.hanning(N_FFT).astype(np.float32)
    n_frames = frames.shape[0]
    n = N_FFT + HOP * (n_frames - 1)
    out = np.zeros(n, dtype=np.float32)
    win_sum = np.zeros(n, dtype=np.float32)
    for t in range(n_frames):
        s = t * HOP
        out[s:s + N_FFT] += frames[t] * win
        win_sum[s:s + N_FFT] += win ** 2
    win_sum[win_sum < 1e-8] = 1e-8
    return out / win_sum


def clip_labels(session_dir):
    """clip 文件名 → 标注串。"""
    u = session_dir / "labeling" / "user_labels.csv"
    if not u.exists():
        return {}
    return {r["clip"]: (r.get("label") or "")
            for r in csv.DictReader(open(u, encoding="utf-8-sig"))}


def collect_specs(session_dir, want):
    """返回 [(clip名, mag, phase)]，只要标注串含 want 的片段。"""
    lab = clip_labels(session_dir)
    out = []
    cdir = session_dir / "labeling" / "clips"
    if not cdir.exists():
        return out
    for f in sorted(cdir.iterdir()):
        tag = lab.get(f.name, "")
        if want in tag:
            x = fix_len(load_clip_wav(f))
            mag, phase = stft_mag(x)
            out.append((f.name, mag, phase, x))
    return out


def build_template(specs):
    """逐片段按能量归一后取逐元素中位数 → 形状模板 T（单位能量）。"""
    norms = []
    for _, mag, _, _ in specs:
        g = np.sqrt((mag ** 2).sum()) + 1e-9
        norms.append(mag / g)
    T = np.median(np.stack(norms), axis=0)
    T /= np.sqrt((T ** 2).sum()) + 1e-9
    return T


def best_lag(env_t, env_r):
    """包络互相关找最优时延（FFT 圆周相关），返回 lag（帧）。"""
    n = len(env_t)
    F_t = np.fft.rfft(env_t)
    F_r = np.fft.rfft(env_r)
    cc = np.fft.irfft(F_t * np.conj(F_r), n=n)
    lag = int(np.argmax(cc))
    if lag > n // 2:
        lag -= n
    return lag


def align(T, mag):
    """按能量包络互相关把模板 T 沿时间轴滚到与 mag 对齐。"""
    lag = best_lag(mag.sum(axis=1), T.sum(axis=1))
    return np.roll(T, lag, axis=0), lag


def subtract(mag, T, alpha):
    Ta, _ = align(T, mag)
    g = float((mag * Ta).sum() / (Ta * Ta).sum())
    resid = np.maximum(mag - alpha * g * Ta, BETA * mag)
    return resid, g


def db(x):
    return 10.0 * np.log10(max(x, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=None,
                    help="指定则只跑该系数并导出 demo 音频")
    ap.add_argument("--held", default=None, help="留出局目录名（默认最新）")
    args = ap.parse_args()

    sessions = sorted(p.name for p in (ROOT / "sessions").glob("2*"))
    held = args.held or sessions[-1]
    train = [s for s in sessions if s != held]
    print(f"模板来源 {len(train)} 局，留出验证 {held}")

    tpl_specs, n_mixed = [], 0
    for s in train:
        for name, mag, phase, x in collect_specs(ROOT / "sessions" / s,
                                                 SELF_TAG):
            # 与枪声/爆炸混合的段不进模板（形状被污染）
            tag = clip_labels(ROOT / "sessions" / s).get(name, "")
            if "枪声" in tag or "爆炸" in tag:
                n_mixed += 1
                continue
            tpl_specs.append((name, mag, phase, x))
    print(f"模板片段 {len(tpl_specs)} 条（剔除枪声/爆炸混合 {n_mixed} 条）")
    if len(tpl_specs) < 5:
        print("模板太少，先多标几局 self:脚步")
        return
    T = build_template(tpl_specs)

    held_dir = ROOT / "sessions" / held
    self_specs = collect_specs(held_dir, SELF_TAG)
    foe_specs = [s for s in collect_specs(held_dir, "脚步")
                 if SELF_TAG not in clip_labels(held_dir).get(s[0], "")
                 and "枪声" not in clip_labels(held_dir).get(s[0], "")]
    print(f"留出局：自己脚步 {len(self_specs)} 条，敌我脚步(无误伤参照) "
          f"{len(foe_specs)} 条")

    alphas = [args.alpha] if args.alpha else [0.5, 1.0, 1.5, 2.0, 3.0]
    rows = []
    for alpha in alphas:
        s_db = [db(((r := subtract(m, T, alpha)[0]) ** 2).sum() / (m ** 2).sum())
                for _, m, _, _ in self_specs]
        f_db = [db(((r := subtract(m, T, alpha)[0]) ** 2).sum() / (m ** 2).sum())
                for _, m, _, _ in foe_specs]
        rows.append((alpha, s_db, f_db))
        print(f"alpha={alpha:<4} 自己段抑制 {np.median(s_db):6.1f} dB "
              f"(n={len(s_db)})  敌我段误伤 {np.median(f_db):+6.1f} dB "
              f"(n={len(f_db)})")

    if args.alpha is not None:
        outdir = ROOT / "deliverables" / "tpl_sub_demo"
        outdir.mkdir(parents=True, exist_ok=True)
        demo = self_specs[:DEMO_N_SELF] + foe_specs[:DEMO_N_FOE]
        # 用 wave 写 16k 16bit mono，避免新依赖
        for name, mag, phase, x in demo:
            resid, g = subtract(mag, T, args.alpha)
            y = istft_mag(resid, phase)
            y = y / (np.abs(y).max() + 1e-9) * 0.9
            tag = "self" if any(name == s[0] for s in self_specs) else "foe"
            for suffix, sig in (("mix", x), ("clean", y)):
                pcm = (np.clip(sig[:38400], -1, 1) * 32767).astype(np.int16)
                with wave.open(str(outdir / f"{tag}_{suffix}_{name}"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SR)
                    w.writeframes(pcm.tobytes())
        meta = {"held": held, "alpha": args.alpha,
                "n_tpl": len(tpl_specs),
                "self_median_db": float(np.median([r for r in rows[0][1]])),
                "foe_median_db": float(np.median([r for r in rows[0][2]]))}
        (outdir / "summary.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"demo 音频 → {outdir}")


if __name__ == "__main__":
    main()

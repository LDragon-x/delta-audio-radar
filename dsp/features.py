"""信号层 · 特征提取：一帧双声道 float32 -> 结构化特征字典。

纯函数，无状态，可离线测试。
"""
import numpy as np


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def band_powers(mono, sr, bands):
    """各频段功率 + 结构占比 + 低频谱平坦度。mono 为单声道帧。"""
    n = len(mono)
    win = np.hanning(n)
    sp = np.abs(np.fft.rfft(mono * win)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    def band_mask(lo, hi):
        return (freqs >= lo) & (freqs < hi)

    def band(lo, hi):
        m = band_mask(lo, hi)
        return float(sp[m].sum() / n)

    full = max(band(20, 12000), 1e-12)
    low = band(*bands["low"])
    mid = band(*bands["mid"])
    high = band(*bands["high"])

    # 低频段谱平坦度：geomean/arithmean。
    # 音乐贝斯是"音高类"声音，谐波集中 → 平坦度趋近 0；
    # 脚步是噪声类瞬态，频谱摊匀 → 平坦度高。这是区分两者的关键特征。
    low_sp = sp[band_mask(*bands["low"])]
    if low_sp.sum() > 1e-12:
        flat = float(np.exp(np.mean(np.log(low_sp + 1e-20))) / (np.mean(low_sp) + 1e-20))
    else:
        flat = 0.0

    return {
        "full": full,
        "low": low,
        "mid": mid,
        "high": high,
        "low_ratio": low / full,
        "mid_ratio": mid / full,
        "high_ratio": high / full,
        "low_flat": flat,
    }


def extract_features(frame, sr, bands):
    """frame: [n, 2]。返回特征字典（含左右 RMS 与声像 pan）。"""
    left, right = frame[:, 0], frame[:, 1]
    mono = (left + right) * 0.5
    feats = band_powers(mono, sr, bands)
    feats["rms_l"] = rms(left)
    feats["rms_r"] = rms(right)
    denom = feats["rms_l"] + feats["rms_r"] + 1e-12
    # pan: -1 全左, +1 全右
    feats["pan"] = float((feats["rms_r"] - feats["rms_l"]) / denom)
    feats["rms"] = rms(mono)
    # 立体声宽度：左右互相关。音乐床/人声几乎单声道 → corr≈1；
    # 游戏 HRTF 双耳流天然去相关 → corr 明显低于 1。剔除单声道假目标的关键特征。
    if feats["rms_l"] > 1e-5 and feats["rms_r"] > 1e-5:
        c = float(np.corrcoef(left, right)[0, 1])
        feats["corr"] = 0.0 if np.isnan(c) else c
    else:
        feats["corr"] = 0.0
    return feats

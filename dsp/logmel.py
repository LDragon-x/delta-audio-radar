# -*- coding: utf-8 -*-
"""v0.2 特征基座：对数梅尔频谱 + 时序池化，纯 numpy（无 librosa 依赖）。

背景（JOURNAL 三次撞墙）：v0.1 的 8 维包络特征对人机/真人脚步、
敌我脚步零区分力（信噪比 1.06）。9-27 试过单步 64 带 log-mel 均值
指纹，40v40 样本下也是掷硬币——所以本模块不做"换个特征 magically 能分"
的承诺，它是特征弹药库：

  - mel_bank(): 梅尔三角滤波器组（HTK 公式）
  - stft_power(): 短时傅里叶功率谱
  - logmel_patch(): 事件窗的 log-mel 时频图（二维，可整段喂序列模型）
  - pooled(): 从 patch 池化成向量——mean / mean+std / 时序调制谱
    （脚步纹理在幅度包络的调制频率里，见 KNN-OC 文献）

验证走 tool/feat_eval.py（特征烘焙赛，留出法，诚实报数）。
"""
import numpy as np

_HTK = True  # HTK mel 公式（slaney 线性区不同，这里统一 HTK）


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_bank(sr, n_fft, n_mels=64, fmin=40.0, fmax=None):
    """返回 (n_mels, n_fft//2+1) 三角滤波器组矩阵。"""
    fmax = fmax or sr / 2
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_freqs)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (fft_freqs - lo) / max(ce - lo, 1e-9)
        dn = (hi - fft_freqs) / max(hi - ce, 1e-9)
        bank[i] = np.clip(np.minimum(up, dn), 0, None)
    return bank


def stft_power(x, n_fft=512, hop=128, window="hann"):
    """x: 1D float。返回 (n_frames, n_fft//2+1) 功率谱。"""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    if window == "hann":
        win = np.hanning(n_fft).astype(np.float32)
    else:
        win = np.ones(n_fft, dtype=np.float32)
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * win
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return spec


def logmel_patch(mono, sr, t_center, pre=0.15, post=0.35,
                 n_fft=512, hop=None, n_mels=64, fmin=40.0, fmax=None):
    """事件 t_center 前后窗的 log-mel 时频图。

    返回 (n_mels, n_frames) float32；窗长按 16k sr 约 0.5s → 60 帧。
    """
    hop = hop or n_fft // 4
    i0 = max(0, int((t_center - pre) * sr))
    i1 = min(len(mono), int((t_center + post) * sr))
    seg = mono[i0:i1]
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    bank = mel_bank(sr, n_fft, n_mels, fmin, fmax)
    power = stft_power(seg, n_fft, hop)          # (frames, freqs)
    mel = power @ bank.T                          # (frames, mels)
    logmel = np.log(mel + 1e-10).T                # (mels, frames)
    return logmel.astype(np.float32)


def pcen(mel, s=0.025, alpha=0.98, delta=2.0, eps=1e-6, r=0.5):
    """PCEN 逐通道能量归一化（Wang et al. 2017, arXiv:1706.03200）。

    mel: (n_frames, n_mels) 线性梅尔能量。
    M = (E / (eps + smooth(E))^alpha + delta)^r - delta^r
    smooth 为沿时间的 IIR 包络: smooth_t = (1-s)*smooth_{t-1} + s*E_t
    作用：抑制通道间响度差、自适应增益，嘈杂环境下的事件突出。
    DCASE 文献对照：Zhao2024 同架构 +12 F1，Liu2024 -9 F1——系统依赖，
    本烘焙赛即实测仲裁。
    """
    E = np.asarray(mel, dtype=np.float32)
    if E.shape[0] == 0:
        return E
    smooth = np.empty_like(E)
    smooth[0] = E[0]
    for t in range(1, E.shape[0]):
        smooth[t] = (1.0 - s) * smooth[t - 1] + s * E[t]
    return (E / (eps + smooth) ** alpha + delta) ** r - delta ** r


def pcen_patch(mono, sr, t_center, pre=0.15, post=0.35,
               n_fft=512, hop=None, n_mels=64, fmin=40.0, fmax=None):
    """事件 t_center 前后窗的 PCEN 时频图，返回 (n_mels, n_frames)。"""
    hop = hop or n_fft // 4
    i0 = max(0, int((t_center - pre) * sr))
    i1 = min(len(mono), int((t_center + post) * sr))
    seg = mono[i0:i1]
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    bank = mel_bank(sr, n_fft, n_mels, fmin, fmax)
    power = stft_power(seg, n_fft, hop)          # (frames, freqs)
    mel = power @ bank.T                          # (frames, mels)
    return pcen(mel).T.astype(np.float32)


def pooled(logmel):
    """池化向量组：mean64 / mean+std128 / 时序调制谱 32。

    时序调制谱：每条 mel 带的幅度包络（沿时间轴）做 FFT，取 2-16Hz
    节奏带的能量合计再对数——人机匀速走=窄峰，真人变速=散。
    """
    m, T = logmel.shape
    mean = logmel.mean(axis=1)
    std = logmel.std(axis=1)
    # 调制谱：包络 = mel 带能量沿时间变化（去均值）后 FFT，
    # 取低频调制区（前 1/8 频点）能量和对数——人机匀速走=窄峰集中，
    # 真人变速/停顿=能量散开，谱形差异比手工 jitter 指标信息全。
    env = logmel - logmel.mean(axis=1, keepdims=True)
    n = int(2 ** np.ceil(np.log2(max(T, 2))))
    spec = np.abs(np.fft.rfft(env, n=n, axis=1))        # (mels, freq)
    k = max(1, spec.shape[1] // 8)
    mod = np.log(spec[:, :k].sum(axis=1) + 1e-10)
    mod32 = mod[:32] if m >= 32 else np.pad(mod, (0, 32 - m))
    return {"mel_mean": mean.astype(np.float32),
            "mel_meanstd": np.concatenate([mean, std]).astype(np.float32),
            "mel_mod32": mod32.astype(np.float32)}

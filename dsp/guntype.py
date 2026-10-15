"""枪声指纹：不求距离，只求"是什么枪"。

指纹 = 攻击瞬态 150ms 内的时间平均对数频谱形状（12 频带 × 左右耳），
L2 归一化后余弦最近邻。枪管长度/制退器/弹药决定频谱形状，
时变包络对对齐误差敏感，时间平均形状则稳得多——这是实测对比后的选择。

指纹库由 tool/guntype.py 标注积累；阈值需在干净样本上标定。
"""
import json
from pathlib import Path

import numpy as np

SR = 48000
WIN_PAD_PRE = 0.05       # 指纹窗口：枪声触发点前留 50ms
WIN_PAD_POST = 0.55      # 触发点后留 550ms（供精调 onset 搜索）
WIN_LEN = int(SR * (WIN_PAD_PRE + WIN_PAD_POST))
ATTACK = 0.15            # 攻击瞬态窗（实测：更短的窗谱分辨率不足反而更不稳）
ATTACK_LEN = int(ATTACK * SR)
BANDS12 = [(30, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000),
           (2000, 4000), (4000, 7000), (7000, 10000), (10000, 14000),
           (14000, 18000), (18000, 22000)]
MATCH_THRESH = 0.35    # 余弦距离阈值（起始值，干净样本上标定后收紧）


def normalize_window(mono, sr=SR):
    """把任意长度单声道片段对齐到固定指纹窗口（前 50ms + 后 550ms）。"""
    mono = np.asarray(mono, dtype=np.float32)
    if len(mono) >= WIN_LEN:
        return mono[:WIN_LEN]
    return np.pad(mono, (0, WIN_LEN - len(mono)))


def refine_onset(win, sr=SR):
    """在窗口前 200ms 内按 2ms 跳找能量陡增点，返回窗内样本偏移。"""
    seg = win[:int(0.2 * sr)]
    hop = int(0.002 * sr)
    if len(seg) < hop * 4:
        return 0
    e = np.array([np.sum(seg[i:i + hop] ** 2) for i in range(0, len(seg) - hop, hop)])
    base = np.median(e[:5]) + 1e-9
    return int(np.argmax(e > base * 4)) * hop


def _band_shape(seg, sr=SR):
    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    fr = np.fft.rfftfreq(len(seg), 1 / sr)
    return np.array([np.log(sp[(fr >= lo) & (fr < hi)].sum() + 1e-9)
                     for lo, hi in BANDS12])


def fingerprint(win, attack=None):
    """win: [N, 2] 指纹窗口（起点=触发点-50ms）。返回归一化指纹向量。

    左右耳各算 12 频带时间平均形状后拼接——双耳既保留了更多频谱信息，
    也对单耳录制退化有一定容忍。

    attack 可指定攻击瞬态窗长（秒）：枪型识别用默认 0.15；
    自身声纹实时匹配用 0.08（决策时刻缓冲里必然已有这么长的音频）。
    """
    attack_len = int((attack if attack is not None else ATTACK) * SR)
    win = np.asarray(win, dtype=np.float32)
    if win.ndim == 1:
        win = np.stack([win, win], axis=1)
    s = refine_onset(win.mean(axis=1))
    feats = []
    for ch in range(2):
        seg = win[s:s + attack_len, ch]
        if len(seg) < attack_len // 2:  # 数据不足就退化用全窗
            seg = win[:, ch]
        feats.append(_band_shape(seg))
    v = np.concatenate(feats)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / (n + 1e-12)).round(4).tolist()


def cosine_dist(a, b):
    return float(1.0 - np.dot(a, b))


class GunLibrary:
    """指纹库：枪名 -> 多条指纹（同一枪多环境各存一条，匹配取最近）。"""

    def __init__(self, path="samples/guns/library.json"):
        self.path = Path(path)
        self.guns = {}
        if self.path.exists():
            self.guns = json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, name, fp):
        self.guns.setdefault(name, []).append(fp)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.guns, ensure_ascii=False, indent=1),
                             encoding="utf-8")

    def match(self, fp):
        """返回 (枪名, 距离) 或 (None, 最近距离)。"""
        best_name, best_d = None, 1e9
        for name, fps in self.guns.items():
            for f in fps:
                d = cosine_dist(fp, f)
                if d < best_d:
                    best_name, best_d = name, d
        if best_d > MATCH_THRESH or best_name is None:
            return None, best_d
        return best_name, best_d


class GunTyper:
    def __init__(self, lib_path="samples/guns/library.json"):
        self.lib = GunLibrary(lib_path)

    def identify(self, win_stereo):
        if not self.lib.guns:
            return None, None
        return self.lib.match(fingerprint(win_stereo))

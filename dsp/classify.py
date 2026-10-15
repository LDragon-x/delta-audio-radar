"""信号层 · 事件分类：脚步 / 枪声，以及人机指纹过滤插槽。

BotFingerprint 是预留插槽：v0.1 未训练，恒返回 0.0（不排除）。
v0.2 计划：录人机脚步样本做指纹库（素材统一，模板匹配即可），
替换 is_bot() 的实现，下游代码零改动。

KnnGuess：learn.py 训练的 k-NN 模型接入点（默认 samples/model/knn.json）。
v1 实测结论（12252 个真实事件回放）：8 维块级频带特征分不开真人/人机
脚步，高置信人机判定仅 4 条——所以只作 advisory：bot_suspicion >=0.9
才并入 bot 概率（保守到几乎不触发），预测类写进事件 knn 字段供回放
分析。等特征升级（log-mel 纹理）+ 真人脚步样本增多后再放开。
"""
import json
from pathlib import Path

import numpy as np


class KnnGuess:
    """k-NN 模型（learn.py 产物）。guess 返回 (预测类, 置信度)。"""
    FEAT_KEYS = ["low", "mid", "high", "low_ratio", "high_ratio",
                 "corr", "low_flat", "rms"]
    LOG_KEYS = {"low", "mid", "high", "rms"}

    def __init__(self, path):
        self.ok = False
        try:
            m = json.loads(Path(path).read_text(encoding="utf-8"))
            self.X = np.array(m["X"])
            self.y = m["y"]
            self.mu = np.array(m["mu"])
            self.sd = np.array(m["sd"])
            self.k = m.get("k", 5)
            self.ok = len(self.y) > 0
        except Exception:
            self.ok = False

    def guess(self, feats):
        if not self.ok:
            return None, 0.0
        v = np.array([np.log(feats[k] + 1e-9) if k in self.LOG_KEYS
                      else float(feats[k]) for k in self.FEAT_KEYS])
        d = np.linalg.norm(self.X - (v - self.mu) / self.sd, axis=1)
        kk = np.argsort(d)[:self.k]
        votes = {}
        for j in kk:
            votes[self.y[j]] = votes.get(self.y[j], 0) + 1.0 / (d[j] + 1e-6)
        best = max(votes, key=votes.get)
        return best, votes[best] / sum(votes.values())


class BotFingerprint:
    """人机脚步指纹。score 返回 0.0~1.0 的疑似概率，>=0.5 视为人机剔除。"""

    def score(self, feats, ctx):
        # TODO(v0.2): 录制人机脚步样本，提取模板指纹后启用
        return 0.0


class Classifier:
    def __init__(self, mono_corr_max=0.95, gun_full_min=1e-3, gun_high_min=1e-5,
                 glass_high_ratio=0.30, glass_high_mid=2.0, knn=None):
        self.mono_corr_max = mono_corr_max
        self.gun_full_min = gun_full_min
        self.gun_high_min = gun_high_min
        self.glass_high_ratio = glass_high_ratio
        self.glass_high_mid = glass_high_mid
        self.bot = BotFingerprint()
        self.knn = knn

    def update(self, feats, t):
        """返回事件 dict 或 None。feats 需含 attack（引擎每块预计算）。"""
        attack = feats.get("attack", 1.0)
        quiet = feats.get("quiet", False)

        # 碎玻璃：高频压过中频 2 倍以上且高频占比高（官方可闻 52 米）。
        # 排在枪声前面——玻璃是高通信号，枪声是宽带信号，用 高频/中频 比分开。
        if feats["high"] > feats["mid"] * self.glass_high_mid \
                and feats["high"] > feats["low"] \
                and feats["high_ratio"] > self.glass_high_ratio:
            return {"type": "glass", "t": t, "level": feats["high"],
                    "pan": feats["pan"], "bot": 0.0}

        # 枪声：高频压过低频、攻击陡峭、绝对能量足够（能量地板防随机脉冲误报）
        if feats["high"] > feats["low"] and feats["high_ratio"] > 0.20 \
                and attack > 3.0 and feats["full"] > self.gun_full_min \
                and feats["high"] > self.gun_high_min:
            return {"type": "gunshot", "t": t, "level": feats["full"],
                    "pan": feats["pan"], "bot": 0.0}

        # 脚步：低频主导。静步档事件跳过单声道门限——它们本来就低置信，
        # 而在相关噪声床（如低通风噪）里安静脚步会被门限误杀，宁可多报。
        if feats["low"] > feats["high"] * 2.0 and \
                (quiet or feats["corr"] < self.mono_corr_max):
            bot_prob = self.bot.score(feats, {"t": t})
            knn_tag = None
            if self.knn is not None and self.knn.ok:
                knn_cls, knn_conf = self.knn.guess(feats)
                if knn_cls:
                    knn_tag = f"{knn_cls}:{knn_conf:.2f}"
                    # 高置信人机判定才抬升 bot 概率（v1 特征保守，见模块注释）
                    if knn_cls == "bot_footstep":
                        bot_prob = max(bot_prob, knn_conf)
            if bot_prob < 0.5:
                ev = {"type": "footstep", "t": t, "level": feats["low"],
                      "pan": feats["pan"], "bot": bot_prob, "quiet": quiet}
            else:
                ev = {"type": "bot_footstep", "t": t, "level": feats["low"],
                      "pan": feats["pan"], "bot": bot_prob}
            if knn_tag:
                ev["knn"] = knn_tag
            return ev

        # 行为声（换弹/打药/缠绷带/开关门等中频金属瞬态族）：
        # 中频主导、非枪非玻璃。子类区分（换弹 vs 打药）需 v0.2 样本训练，
        # 现在先报到"行为声"这一级——按官方表，听到即知 29 米内有人。
        if feats["mid"] > feats["low"] * 1.2 and feats["mid"] > feats["high"]:
            return {"type": "action", "t": t, "level": feats["mid"],
                    "pan": feats["pan"], "bot": 0.0}
        return None


class GunshotMasking:
    """枪声后的短暂窗口内压制脚步检测——这段时间信息已被物理摧毁，硬算只出误报。"""

    def __init__(self, cfg):
        self.suppress = cfg["gunshot_suppress_ms"] / 1000.0
        self.until = -1e9

    def suppresses(self, event):
        """枪声本身照常上报，但打开压制窗口；窗口内的脚步事件被丢弃。"""
        if event is None:
            return False
        if event["type"] == "gunshot":
            self.until = event["t"] + self.suppress
            return False
        return event["t"] < self.until

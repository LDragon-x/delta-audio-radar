"""信号层 · 脚步触发：低频段能量相对自适应底噪的突增检测。

核心思想：触发不看绝对阈值，看"相对当前环境底噪的突增"——
枪声再吵只是抬高底噪，脚步的相对突起仍在。这是嘈杂环境定向的第一道关。
"""


class OnsetDetector:
    """瞬态触发：双档位。

    强档：低频/全频 surge 远超底噪（脚步、枪声）。
    静步档：低频 surge 只超底噪 ~1.8 倍也放行——对应官方数据里
    卧倒/蹲静步 6-9 米级的低响度移动声（samples/reference/sound-distances.md），
    代价是置信度低，事件带 quiet 标记，由 tracker/UI 决定是否展示。
    频段归属交给 Classifier，这里只回答"有东西响了、响得多轻"。
    """

    def __init__(self, cfg):
        self.ratio = cfg["floor_ratio"]
        self.quiet_ratio = cfg.get("quiet_ratio", 1.8)
        self.min_low = cfg["min_low"]
        self.min_full = cfg.get("min_full", 1e-6)
        self.min_high = cfg.get("min_high", 1e-6)
        self.quiet_low_ratio_min = cfg.get("quiet_low_ratio_min", 0.25)
        self.cooldown = cfg["cooldown_ms"] / 1000.0
        self.alpha = cfg["floor_alpha"]
        self.floor_low = None
        self.floor_full = None
        self.floor_high = None
        self.last_onset = -1e9

    def update(self, feats, t):
        low, full, high = feats["low"], feats["full"], feats["high"]
        if self.floor_low is None:
            self.floor_low, self.floor_full = low, full
            self.floor_high = high
        hit_low = low > max(self.floor_low * self.ratio, self.min_low)
        hit_full = full > max(self.floor_full * self.ratio * 2, self.min_full)
        # 高频 surge 通道：碎玻璃等高频事件的独立触发线，不依赖低频
        hit_high = high > max(self.floor_high * self.ratio * 2, self.min_high)
        level = None
        if hit_low or hit_full or hit_high:
            level = "strong"
        elif low > max(self.floor_low * self.quiet_ratio, self.min_low * 0.3) \
                and feats["low_ratio"] > self.quiet_low_ratio_min:
            level = "quiet"
        if level and (t - self.last_onset) <= self.cooldown:
            level = None
        if level:
            self.last_onset = t
            return level
        # 非触发期底噪缓慢跟随
        if not hit_low:
            self.floor_low = (1 - self.alpha) * self.floor_low + self.alpha * low
        if not hit_full:
            self.floor_full = (1 - self.alpha) * self.floor_full + self.alpha * full
        if not hit_high:
            self.floor_high = (1 - self.alpha) * self.floor_high + self.alpha * high
        return None

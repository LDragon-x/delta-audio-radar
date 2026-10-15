"""状态层：单目标 EMA 平滑 + 在场保持 + 距离档 + 上下提示预留。

单帧检测全是噪声，这层把事件攒成"一个目标"：
- 角度 EMA 平滑（脚尖抖→稳定指向）
- presence TTL（脚步停 1.5s 内目标仍在）
- 距离档：脚步按官方传播距离分 近≤15m / 中≤32m / 远≤49m；
  行为声直接挂官方可闻距离上界（听到=人在该距离内）
- elevation：v0.2 HRTF 标定后启用，先返回 None（绝不瞎报）
"""


class Tracker:
    def __init__(self, cfg, prox_cfg, ceilings=None):
        self.ema = cfg["ema"]
        self.ttl = cfg["presence_ttl_ms"] / 1000.0
        self.angle_max = cfg["angle_max_deg"]
        self.prox_cfg = prox_cfg
        self.ceilings = ceilings or {}
        self.tier_meters = {"近": 15, "中": 32, "远": 49}  # 官方移动声音传播距离
        self.angle = None
        self.presence_until = -1e9
        self.last_tier = "-"
        self.last_event = None

    def update(self, event, t):
        if event and event["type"] in ("footstep", "action", "glass") \
                and not event.get("self"):
            ang = event["pan"] * self.angle_max
            self.angle = ang if self.angle is None else (1 - self.ema) * self.angle + self.ema * ang
            self.presence_until = t + self.ttl
            self.last_tier = self._tier(event)
            self.last_event = event
        return self.snapshot(t)

    def _tier(self, event):
        if event["type"] in self.ceilings:
            return f"≤{self.ceilings[event['type']]}m"
        level = event["level"]
        if level >= self.prox_cfg["near_low"]:
            return f"近·≤{self.tier_meters['近']}m"
        if level >= self.prox_cfg["mid_low"]:
            return f"中·≤{self.tier_meters['中']}m"
        return f"远·≤{self.tier_meters['远']}m"

    def snapshot(self, t):
        present = t < self.presence_until
        return {
            "present": present,
            "angle": round(self.angle, 1) if (present and self.angle is not None) else None,
            "tier": self.last_tier if present else "-",
            "elevation": None,  # v0.2: 基于 HRTF 频谱标定
        }

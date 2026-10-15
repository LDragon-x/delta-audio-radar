"""自身声音过滤：把自己角色的声音从检测流里踢出去。

设计逻辑（用户洞察：自己的声音永远是最大的）：
第一人称游戏自听音效（脚步/枪声/换弹）有几个稳定特征：
- 音量全场最大（无距离衰减）
- 声像在正中附近（自听 foley 不做空间化，或弱空间化）
- 种类有限（自己干员的动作采样是固定几套）→ 指纹可枚举

三层防线：
  1. 热键掩蔽（即时生效）：自己按换弹/切枪键时， suppress 对应事件
  2. 规则打分（零样本启动）：响度全场分位 + 居中程度 + 无混响尾巴 → self_score
  3. 自身声纹库（玩几局自动攒）：collector 录音里"最响最居中最高频出现"的
     聚类 = 自己的声音指纹 → 模板匹配 suppress。
     指纹机与枪型识别共用（dsp/guntype.py），但攻击窗取 0.08s——
     决策时刻（触发后 ≥0.09s）滚动缓冲里必然已覆盖 t0+0.13s，
     所以同步匹配不需要像枪型识别那样再等 0.55s，输出零额外延迟。
     优先级：热键掩蔽 > 声纹库 > 规则打分。

被过滤的事件不丢弃，打 self=True 标记——它们对"人机/真人"分类器
是绝佳的负样本（自己的声音和敌人声音在特征空间里天然两簇）。
"""
import json
import threading
from pathlib import Path

import numpy as np

from dsp.guntype import fingerprint, cosine_dist

VP_PRE = 0.05    # 声纹窗口：触发点前留 50ms（与枪纹同构，供 refine_onset）
VP_POST = 0.13   # 触发点后 130ms：决策时刻缓冲必然已覆盖，几何固定才可复现
VP_ATTACK = 0.08 # 攻击瞬态窗：脚步/换弹 foley 起音 80ms 足够分敌我
VP_MATCH_THRESH = 0.30  # 起始阈值，干净样本上标定后收紧


def vp_window(buf, t0):
    """从引擎滚动缓冲取声纹窗口 [t0-0.05, t0+0.13]（[N, 2]）。

    决策发生在触发后 ≥0.09s，块边界保证音频覆盖到 t0+0.13s 以上，
    所以这个窗口在决策时刻必然完整——与建库时从 wav 截取同一几何，
    线上线下指纹对齐才一致。
    """
    parts = [fr for bt, fr in buf if t0 - VP_PRE <= bt <= t0 + VP_POST]
    if not parts:
        return None
    return np.concatenate(parts, axis=0)


class SelfVoiceprint:
    """防线3：自身声纹库。指纹命中即标 self，理由 voiceprint:<类别>。

    库文件：{类别: [指纹向量, ...]}，由 tool/selfvoice.py 从 collector
    会话（自标标签 + 规则已标 self 的事件）攒出来。
    """

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.thresh = float(cfg.get("thresh", VP_MATCH_THRESH))
        self.lib_path = Path(cfg.get("lib", "samples/self/library.json"))
        self.lib = {}
        if self.enabled and self.lib_path.exists():
            self.lib = json.loads(self.lib_path.read_text(encoding="utf-8"))

    def check(self, win):
        """命中返回 (类别, 距离)，未命中返回 None。库空时不拖慢链路。"""
        if not self.lib or win is None:
            return None
        fp = fingerprint(win, attack=VP_ATTACK)
        best_cat, best_d = None, 1e9
        for cat, fps in self.lib.items():
            for f in fps:
                d = cosine_dist(fp, f)
                if d < best_d:
                    best_cat, best_d = cat, d
        if best_cat is not None and best_d <= self.thresh:
            return best_cat, best_d
        return None


# 常用键的虚拟键码（GetAsyncKeyState 用）。表外的键不支持按住检测，
# 但仍可走 press 窗口路径。
_VK = {"left": 0x01, "right": 0x02, "middle": 0x04,
       "space": 0x20, "shift": 0x10, "ctrl": 0x11, "alt": 0x12,
       "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34, "5": 0x35,
       "a": 0x41, "b": 0x42, "c": 0x43, "d": 0x44,
       "q": 0x51, "r": 0x52, "s": 0x53, "w": 0x57,
       "x": 0x58, "z": 0x5A}


class HotkeyMask:
    """防线1（即时生效）：自己在游戏里按换弹/切枪/移动/开火键时，抑制对应事件。

    键盘回调跑在独立线程，时间轴统一用引擎的音频时间 t——
    Engine.on_hotkey(key) 在收到按键时以当前音频时间开窗。
    window 留有余量（默认 2.5s），吸收采集队列的回放延迟，
    宁可多标自己（负样本无害）也不少标（雷达误报敌人）。

    两种掩蔽语义：
    - press 窗口：按键后 window 秒内标 self（换弹/打药这种一次性的动作声）；
    - hold 按住：hold_labels 里的键（移动/开火）用 GetAsyncKeyState 直读
      物理按键状态，按住期间持续标 self，松手只剩 window 短尾。
      不用 keyboard.is_pressed——它依赖钩子状态跟踪，本机实测恒 False；
      GetAsyncKeyState 是 Win32 物理直读，无消息循环依赖，键盘鼠标同一 API。
    """

    def __init__(self, cfg=None):
        cfg = cfg or {}
        keys = dict(cfg.get("keys", {}))
        # 鼠标键并入同一查表，加前缀避免和键盘键撞名（_ 开头是注释键，跳过）
        for btn, spec in (cfg.get("mouse") or {}).items():
            if isinstance(spec, dict) and not btn.startswith("_"):
                keys["mouse:" + btn.lower()] = spec
        self.keys = {k.lower(): v for k, v in keys.items()}
        self.hold_labels = set(cfg.get("hold_labels", ("move", "fire")))
        # 闩锁：点按型移动（用户是极速点按微操，按住直读经常扑空）后
        # latch_s 内持续标 self，按键可重复触发顺延。宁可多标自己（负样本
        # 无害）也不少标（雷达误报敌人）。
        self.latch_labels = set(cfg.get("latch_labels", ("move",)))
        self.latch_s = float(cfg.get("latch_s", 1.5))
        self.until = 0.0
        self.latch_until = 0.0
        self.reason = ""
        self._lock = threading.Lock()
        self._getasync = None
        try:
            import ctypes
            self._getasync = ctypes.windll.user32.GetAsyncKeyState
        except Exception:
            pass

    def _phys_pressed(self, name):
        """Win32 物理按键直读（键盘鼠标通用，0x8000=高位按下的意思）。"""
        if self._getasync is None:
            return False
        vk = _VK.get(name)
        return bool(vk and self._getasync(vk) & 0x8000)

    def press(self, key, t_now):
        """返回是否命中配置（没配置的键不掩蔽）。"""
        spec = self.keys.get(key.lower())
        if not spec:
            return False
        with self._lock:
            self.until = t_now + float(spec.get("window", 2.0))
            if spec.get("label") in self.latch_labels:
                self.latch_until = t_now + self.latch_s
            self.reason = spec.get("label", key)
        return True

    def active(self, t):
        with self._lock:
            if t <= self.until or t <= self.latch_until:
                return True
        # hold_labels 的键：按住期间持续掩蔽（物理直读，松手即停）
        for k, spec in self.keys.items():
            if spec.get("label") not in self.hold_labels:
                continue
            name = k[6:] if k.startswith("mouse:") else k
            if self._phys_pressed(name):
                with self._lock:
                    self.reason = spec.get("label", k)
                return True
        return False

    def state(self, t):
        with self._lock:
            if t > self.until:
                return None
            return {"reason": self.reason, "until": round(self.until, 2),
                    "remaining": round(self.until - t, 2)}


class SelfSoundFilter:
    def __init__(self, level_p90=None, pan_center=0.20, loud_ratio=3.0):
        self.level_p90 = level_p90      # 历史响度 90 分位（首次调用时自适应）
        self.pan_center = pan_center    # 居中阈值
        self.loud_ratio = loud_ratio    # 超过响度分位多少倍算"自己级别响"
        self._levels = []

    def _update_level(self, feats):
        self._levels.append(feats["rms"])
        if len(self._levels) > 2000:
            self._levels = self._levels[-2000:]
        if len(self._levels) >= 50:
            self.level_p90 = sorted(self._levels)[int(len(self._levels) * 0.9)]

    def score(self, feats, event):
        """返回 0~1 的 self 置信度。

        单声道是必要条件不是加分项：HRTF 空间音频里敌人声音必然去相关，
        只有自听 2D 音效近单声道。响+居中但去相关 = 正面敌人，不能误杀。
        """
        if feats["corr"] <= 0.85:
            return 0.0
        s = 0.3  # 近单声道基础分
        # 响度：全场最大声的候选
        if self.level_p90 and feats["rms"] > self.level_p90 * self.loud_ratio:
            s += 0.4
        # 居中：自听音效不做空间化
        if abs(feats["pan"]) < self.pan_center:
            s += 0.3
        return min(1.0, s)

    def is_self(self, feats, event):
        sc = self.score(feats, event)
        if sc >= 0.6:
            event = dict(event)
            event["self"] = True
            event["self_score"] = round(sc, 2)
            return event
        event = dict(event)
        event["self"] = False
        return event

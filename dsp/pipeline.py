"""引擎：源 -> 特征 -> 触发 -> 分类/掩蔽 -> 自身过滤 -> 状态，逐块流转。

处理链每一环都是纯数据结构，UI 层只消费 snapshot，不参与计算。

触发后不立即分类，而是缓存 90ms 窗口、取低频峰值帧再分类——
触发往往落在起音斜坡，直接分类要么误拒要么吃掉冷却。

枪型识别延迟 0.55s 出结果：需要枪声尾巴（制退器声、机械循环声）才能分枪，
拿到完整 0.6s 窗口后做指纹最近邻匹配。

自身声音过滤（dsp/selfsound.py）：自己的脚步/行为声最响最居中，
标记 self=True——雷达只显示敌人，自己的声音进负样本库。
"""
from collections import deque

import numpy as np

from dsp.features import extract_features
from dsp.onset import OnsetDetector
from dsp.classify import Classifier, GunshotMasking, KnnGuess
from dsp.guntype import GunTyper
from dsp.selfsound import SelfSoundFilter, HotkeyMask, SelfVoiceprint, vp_window
from state.tracker import Tracker


class Engine:
    DECISION_DELAY = 0.09  # 秒
    GUN_ID_DELAY = 0.55    # 秒：等枪声尾巴进缓冲再识别

    def __init__(self, cfg):
        self.cfg = cfg
        self.onset = OnsetDetector(cfg["onset"])
        cc = cfg["classify"]
        self.classifier = Classifier(cc["mono_corr_max"], cc.get("gun_full_min", 1e-3),
                                     cc.get("gun_high_min", 1e-5),
                                     cc.get("glass_high_ratio", 0.30),
                                     cc.get("glass_high_mid", 2.0),
                                     knn=KnnGuess(cc["knn_model"])
                                     if cc.get("knn_model") else None)
        self.masking = GunshotMasking(cfg["masking"])
        self.tracker = Tracker(cfg["tracker"], cfg["proximity"], cfg.get("distance_ceilings"))
        self.guntyper = GunTyper()
        self.selffilter = SelfSoundFilter()
        hm = cfg.get("hotkey_mask") or {}
        self.hotkey_mask = HotkeyMask(hm if hm.get("enabled", True) else {})
        self.voiceprint = SelfVoiceprint(cfg.get("self_voiceprint"))
        self._t = 0.0
        self.prev_full = 0.0
        self._pending = None
        self._pending_gun = None
        self.buf = deque(maxlen=48)  # (t, frame) 滚动缓冲，供枪纹窗口取用

    def on_hotkey(self, key):
        """全局热键回调（keyboard 库线程）：命中配置则开掩蔽窗。

        音频时间 t 由 process 持续刷新；窗口余量见 HotkeyMask 注释。
        """
        return self.hotkey_mask.press(key, self._t)

    def on_mouse(self, button):
        """全局鼠标键回调（mouse 库线程）：机制同 on_hotkey。"""
        return self.hotkey_mask.press("mouse:" + button.lower(), self._t)

    def process(self, t, frame):
        """frame: [n, 2] float32。返回该块的 snapshot（含特征，供调试面板显示）。"""
        self._t = t
        self.buf.append((t, frame))
        feats = extract_features(frame, self.cfg["sample_rate"], self.cfg["bands"])
        feats["attack"] = feats["full"] / max(self.prev_full, 1e-12)
        self.prev_full = feats["full"]
        self.selffilter._update_level(feats)  # 响度分位每块更新，不能只在事件时

        level = self.onset.update(feats, t)
        if level:
            self._pending = {"t": t, "frames": [(t, feats)], "quiet": level == "quiet"}
        elif self._pending is not None:
            self._pending["frames"].append((t, feats))

        event = None
        if self._pending is not None and t - self._pending["t"] >= self.DECISION_DELAY:
            peak_t, peak_feats = max(self._pending["frames"], key=lambda x: x[1]["low"])
            peak_feats = dict(peak_feats)
            peak_feats["quiet"] = self._pending["quiet"]
            event = self.classifier.update(peak_feats, self._pending["t"])
            # 自身声音的漏网通道：近单声道+全场最响+低频主导——
            # 这类声音过不了 corr 门限，但它恰恰就是自己的脚步
            if event is None and peak_feats.get("corr", 0) > 0.9 \
                    and self.selffilter.level_p90 \
                    and peak_feats["rms"] > self.selffilter.level_p90 * 2 \
                    and peak_feats["low"] > peak_feats["high"] * 2:
                event = {"type": "footstep", "t": self._pending["t"],
                         "level": peak_feats["low"], "pan": peak_feats["pan"],
                         "bot": 0.0, "quiet": False}
            if self.masking.suppresses(event):
                event = None  # 枪声掩蔽窗口内，脚步信息不可信，丢弃
            elif event is not None and event["type"] in ("footstep", "action", "glass"):
                event = self.selffilter.is_self(peak_feats, event)
            # 防线3 声纹库：命中即标 self。枪声也查——自己的枪声是全链路上
            # 最后一个没被过滤的自听声源。窗口在决策时刻必然完整（见 vp_window）。
            if event is not None and self.voiceprint.enabled \
                    and event["type"] in ("footstep", "action", "glass", "gunshot"):
                hit = self.voiceprint.check(vp_window(self.buf, event["t"]))
                if hit:
                    cat, d = hit
                    event = dict(event)
                    event["self"] = True
                    event["self_reason"] = "voiceprint:" + cat
                    event["self_vp_dist"] = round(d, 3)
            if event is not None and self.hotkey_mask.active(event["t"]):
                # 热键掩蔽窗内的事件一律是自己（换弹/切枪声被引擎误判时纠正），
                # 与规则层共存：热键优先，理由覆盖
                event = dict(event)
                event["self"] = True
                event["self_reason"] = "hotkey:" + self.hotkey_mask.reason
            self._pending = None
            if event is not None and event["type"] == "gunshot":
                self._pending_gun = {"t": event["t"]}

        snap = self.tracker.update(event, t)
        snap["feats"] = feats
        snap["floor"] = self.onset.floor_low
        snap["masked"] = t < self.masking.until
        snap["hotkey_mask"] = self.hotkey_mask.state(t)
        snap["event"] = event
        snap["gun_id"] = None

        if self._pending_gun is not None and \
                t - self._pending_gun["t"] >= self.GUN_ID_DELAY:
            t0 = self._pending_gun["t"]
            parts = [fr for bt, fr in self.buf if t0 - 0.05 <= bt <= t0 + 0.55]
            if parts:
                win = np.concatenate(parts, axis=0)  # [N, 2] 保留双耳
                name, d = self.guntyper.identify(win)
                snap["gun_id"] = {"name": name, "dist": None if d is None else round(d, 3),
                                  "t": round(t0, 2)}
            self._pending_gun = None

        snap["t"] = t
        return snap

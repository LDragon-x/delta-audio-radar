"""样本采集器：loopback 录音 + 全局热键打标签 + 引擎自动检测存档。

为 v0.2 标定攒数据：人机指纹、距离档、HRTF 仰角都需要带标签的真机样本。

用法：
  python tool/collector.py                     # 录默认 loopback（本机=入梦虚拟线）
  python tool/collector.py --device "GAMING"   # 显式指定（注意：本机耳机 loopback 不送数据）
  python tool/collector.py --seconds 300       # 录 5 分钟自动停（测试用）

热键（全局生效，游戏前台也能按）：
  F1 真人脚步   F2 人机脚步   F3 枪声
  F4 楼上       F5 楼下      F6 同层
  F7 近         F8 远
  F10 / Ctrl+C  停止并保存

输出 sessions/<时间戳>/：
  session.wav      原始音频（16bit 48kHz 立体声）
  video.mp4        屏幕录像（默认10fps标称，实际约7fps受截屏吞吐限制；时间对位用 video_frames.json，别看 mp4 时长）
  keys.jsonl       按键流（t/键/按下抬起边沿，喂模型用，--no-keys 关闭）
  labels.jsonl     人工标签，每行 {"t": 秒, "label": 名称}
  auto_events.jsonl 引擎自动检测事件（同一份音频的人机对照组）
  report.txt       人工 vs 自动的对账摘要（±0.5s 窗口算命中）
"""
import argparse
import json
import sys
import threading
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.source import LoopbackSource, list_loopback_devices
from dsp.pipeline import Engine

LABELS = {
    "f1": "footstep",
    "f2": "bot_footstep",
    "f3": "gunshot",
    "f4": "above",
    "f5": "below",
    "f6": "same_floor",
    "f7": "near",
    "f8": "far",
    "r": "self_reload",
    "b": "self_firemode",
}

MATCH_WINDOW = 0.5  # 人工标签与自动事件的时间对齐窗口（秒）


class SessionRecorder:
    def __init__(self, cfg, device_keyword=None, outdir="sessions",
                 video=True, video_fps=10, keys_on=True):
        self.cfg = cfg
        self.src = LoopbackSource(device_keyword=device_keyword,
                                  sr=cfg["sample_rate"], block=cfg["block_size"])
        self.engine = Engine(cfg)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.dir = Path(outdir) / ts
        self.dir.mkdir(parents=True, exist_ok=True)
        self.wav_path = self.dir / "session.wav"
        self._wav = wave.open(str(self.wav_path), "wb")
        self._wav.setnchannels(2)
        self._wav.setsampwidth(2)
        self._wav.setframerate(cfg["sample_rate"])
        self.labels = []
        self.auto = []
        self.running = True
        self.t0 = None
        self._lock = threading.Lock()
        self.video = None
        self.video_on = video
        self.video_fps = video_fps
        self.keys = None
        self.keys_on = keys_on

    def label(self, name):
        if self.t0 is None:
            return
        t = time.perf_counter() - self.t0
        with self._lock:
            self.labels.append({"t": round(t, 3), "label": name})
        print(f"\r[标签] {name:<14} @ {t:8.1f}s" + " " * 20)

    def stop(self):
        self.running = False

    def run(self, seconds=None):
        import numpy as np
        self.t0 = time.perf_counter()
        if self.video_on:
            from capture.screen import ScreenRecorder
            self.video = ScreenRecorder(self.dir / "video.mp4",
                                        anchor=lambda: self.src._t0,
                                        fps=self.video_fps)
            self.video.start()
        print(f"录音设备: {self.src.device_name}")
        if self.video_on:
            print(f"屏幕录制: 开（{self.video_fps}fps, video.mp4）")
        hk = self.cfg.get("hotkey_mask") or {}
        if getattr(self, "keys_on", True):
            from capture.keys import KeyLogger
            names = [k for k in hk.get("keys", {}) if not k.startswith("_")]
            names += ["mouse:" + b for b, s in (hk.get("mouse") or {}).items()
                      if isinstance(s, dict) and not b.startswith("_")]
            ml = [n[6:] if n.startswith("mouse:") else n for n in names]
            self.keys = KeyLogger(self.dir / "keys.jsonl",
                                  t0_fn=lambda: self.t0, keys=ml,
                                  mouse_path=self.dir / "mouse.jsonl")
            self.keys.start()
            print(f"键鼠记录: 开（{len(ml)} 个键, keys.jsonl + mouse.jsonl）")
        print(f"输出目录: {self.dir}")
        print("热键: " + "  ".join(f"{k.upper()}={v}" for k, v in LABELS.items()))
        print("F10 或 Ctrl+C 停止\n")
        try:
            while self.running:
                item = self.src.read()
                if item is None:
                    break
                t, frame = item
                pcm = (np.clip(frame, -1, 1) * 32767).astype(np.int16)
                self._wav.writeframes(pcm.tobytes())
                snap = self.engine.process(t, frame)
                ev = snap.get("event")
                if ev and ev["type"] in ("footstep", "gunshot", "bot_footstep",
                                          "glass", "action"):
                    f = snap["feats"]
                    self.auto.append({"t": round(ev["t"], 3), "type": ev["type"],
                                      "pan": round(ev["pan"], 3),
                                      "self": bool(ev.get("self")),
                                      "self_reason": ev.get("self_reason", ""),
                                      "feat": {k: round(float(f[k]), 6) for k in
                                               ("low", "mid", "high", "low_ratio",
                                                "high_ratio", "corr", "low_flat",
                                                "rms")}})
                    side = "左" if ev["pan"] < -0.35 else ("右" if ev["pan"] > 0.35 else "中")
                    print(f"[自动] {ev['type']:<12} {side} t={ev['t']:8.1f}s")
                if seconds and time.perf_counter() - self.t0 >= seconds:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self._finalize()

    def _finalize(self):
        self.src.close()
        self._wav.close()
        if self.video is not None:
            err = self.video.stop()
            vpath = self.dir / "video.mp4"
            if err:
                print(f"[!] {err}")
            elif vpath.exists():
                print(f"视频已存: {vpath.name}（{self.video.frames} 帧）")
        if self.keys is not None:
            err = self.keys.stop()
            kpath = self.dir / "keys.jsonl"
            if err:
                print(f"[!] {err}")
            elif kpath.exists():
                msg = f"键鼠已存: {kpath.name}（{self.keys.edges} 条边沿）"
                if self.keys.mouse_path:
                    msg += (f" + {Path(self.keys.mouse_path).name}"
                            f"（{self.keys.mouse_moves} 条移动）")
                print(msg)
        with open(self.dir / "labels.jsonl", "w", encoding="utf-8") as f:
            for row in self.labels:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(self.dir / "auto_events.jsonl", "w", encoding="utf-8") as f:
            for row in self.auto:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report = self._report()
        with open(self.dir / "report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("\n" + report)
        print(f"已保存: {self.dir}")

    def _report(self):
        human_steps = [x for x in self.labels if x["label"] in ("footstep", "bot_footstep")]
        auto_steps = [x for x in self.auto if x["type"] in ("footstep", "bot_footstep")]
        matched = sum(
            1 for h in human_steps
            if any(abs(a["t"] - h["t"]) <= MATCH_WINDOW for a in auto_steps)
        )
        lines = [
            f"会话时长约 {time.perf_counter() - self.t0:.0f}s",
            f"人工标签 {len(self.labels)} 条，自动事件 {len(self.auto)} 条",
            f"脚步对账（±{MATCH_WINDOW}s 窗口）: 人工 {len(human_steps)}, "
            f"自动命中 {matched} ({matched / max(1, len(human_steps)) * 100:.0f}%)",
            f"  真人脚步标签 {sum(1 for x in self.labels if x['label'] == 'footstep')}, "
            f"人机脚步标签 {sum(1 for x in self.labels if x['label'] == 'bot_footstep')}",
        ]
        by_label = {}
        for x in self.labels:
            by_label[x["label"]] = by_label.get(x["label"], 0) + 1
        lines.append("  标签分布: " + ", ".join(f"{k}={v}" for k, v in sorted(by_label.items())))
        return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="loopback 设备名关键词")
    ap.add_argument("--seconds", type=float, default=None, help="录 N 秒后自动停止")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--outdir", default="sessions")
    ap.add_argument("--no-video", action="store_true", help="关闭屏幕录制")
    ap.add_argument("--no-keys", action="store_true", help="关闭按键记录")
    ap.add_argument("--fps", type=float, default=10, help="录屏帧率（默认 10，标称值，实际受截屏吞吐限制）")
    args = ap.parse_args()

    if args.list:
        for name in list_loopback_devices():
            print(name)
        return

    cfg = json.load(open("config.json", encoding="utf-8"))
    rec = SessionRecorder(cfg, device_keyword=args.device, outdir=args.outdir,
                          video=not args.no_video, video_fps=args.fps,
                          keys_on=not args.no_keys)

    try:
        import keyboard
        for key, name in LABELS.items():
            keyboard.on_press_key(key, lambda _e, n=name: rec.label(n))
        keyboard.on_press_key("f10", lambda _e: rec.stop())
        print("全局热键已挂载")
    except ImportError:
        print("未安装 keyboard 库，退化为控制台打标（输入 1-8 对应标签，回车提交）")
        mapping = {str(i + 1): n for i, n in enumerate(LABELS.values())}

        def console_loop():
            while rec.running:
                line = sys.stdin.readline().strip()
                if line in mapping:
                    rec.label(mapping[line])
                elif line == "q":
                    rec.stop()

        threading.Thread(target=console_loop, daemon=True).start()
    try:
        import mouse
        hm = cfg.get("hotkey_mask") or {}
        btns = {b.lower() for b, s in (hm.get("mouse") or {}).items()
                if isinstance(s, dict) and not b.startswith("_")}

        def _mcb(ev, _btns=btns):
            # mouse.hook 会把移动/滚轮事件也喂进来（MoveEvent 等，不是 dict）
            if not hasattr(ev, "get"):
                return
            if ev.get("event_type") == "down" and ev.get("button", "").lower() in _btns:
                rec.engine.on_mouse(ev["button"])

        if btns:
            mouse.hook(_mcb)
            print("鼠标掩蔽已挂载: " + ", ".join(sorted(btns)))
    except ImportError:
        print("未安装 mouse 库，鼠标掩蔽不可用（pip install mouse）")

    rec.run(seconds=args.seconds)


if __name__ == "__main__":
    main()

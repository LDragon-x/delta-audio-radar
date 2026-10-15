"""调试面板：tkinter 实时显示引擎输出。

只消费 snapshot，不参与计算。左侧雷达 + 状态，右侧特征曲线 + 事件流。
"""
import math
import time
import tkinter as tk

BG = "#10141c"
FG = "#cfd8e3"
ACCENT = "#3fd68f"
WARN = "#e8b13f"
DANGER = "#e85f5f"


class Panel:
    def __init__(self, q, device_name="", show_self=False):
        self.q = q
        self.show_self = show_self  # 输出层过滤：默认不显示自己发出的声音
        self.root = tk.Tk()
        self.root.title(f"Delta Audio Radar v0.1 — {device_name}")
        self.root.configure(bg=BG)
        self.root.geometry("960x460")

        # 左：雷达
        self.radar = tk.Canvas(self.root, width=460, height=440, bg=BG, highlightthickness=0)
        self.radar.pack(side=tk.LEFT, padx=8, pady=8)

        # 右：信息区
        right = tk.Frame(self.root, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.status = tk.Label(right, text="待机", bg=BG, fg=FG,
                               font=("Microsoft YaHei UI", 16, "bold"), anchor="w")
        self.status.pack(fill=tk.X)

        self.sub = tk.Label(right, text="—", bg=BG, fg=FG,
                            font=("Microsoft YaHei UI", 11), anchor="w")
        self.sub.pack(fill=tk.X)

        self.bars = tk.Canvas(right, width=460, height=60, bg=BG, highlightthickness=0)
        self.bars.pack(fill=tk.X, pady=4)

        tk.Label(right, text="低频能量(绿) vs 自适应底噪(灰) — 绿尖峰即脚步触发",
                 bg=BG, fg="#6b7a8d", anchor="w",
                 font=("Microsoft YaHei UI", 9)).pack(fill=tk.X)
        self.scope = tk.Canvas(right, width=460, height=120, bg="#0b0e14", highlightthickness=0)
        self.scope.pack(fill=tk.X, pady=2)

        tk.Label(right, text="事件流", bg=BG, fg="#6b7a8d", anchor="w",
                 font=("Microsoft YaHei UI", 9)).pack(fill=tk.X)
        self.logbox = tk.Listbox(right, height=8, bg="#0b0e14", fg=FG,
                                 highlightthickness=0, font=("Consolas", 9))
        self.logbox.pack(fill=tk.BOTH, expand=True)

        self.hist = []       # (low, floor) 历史
        self.latest = None
        self.last_log_t = 0

    def run(self):
        self.root.after(30, self._tick)
        self.root.mainloop()

    def _tick(self):
        try:
            while True:
                snap = self.q.get_nowait()
                if snap is None:
                    self.status.config(text="源已结束", fg=FG)
                    self.root.after(100, self._redraw)
                    return
                self.latest = snap
                self.hist.append((snap["feats"]["low"], snap["floor"] or 0))
                if len(self.hist) > 160:
                    self.hist.pop(0)
                ev = snap.get("event")
                if ev and ev["type"] in ("footstep", "gunshot") and \
                        (self.show_self or not ev.get("self")) and \
                        ev["t"] - self.last_log_t > 0.2:
                    self.last_log_t = ev["t"]
                    self._log(ev, snap)
        except Exception:
            pass
        self._redraw()
        self.root.after(30, self._tick)

    def _log(self, ev, snap):
        stamp = time.strftime("%H:%M:%S")
        tier = snap["tier"]
        if ev["type"] == "footstep":
            side = self._side(ev["pan"])
            tag = "静步" if ev.get("quiet") else "脚步"
            self.logbox.insert(0, f"{stamp}  {tag}  {side:　<4}  {tier}")
        elif ev["type"] == "gunshot":
            self.logbox.insert(0, f"{stamp}  枪声  {self._side(ev['pan'])}")
        elif ev["type"] == "glass":
            self.logbox.insert(0, f"{stamp}  碎玻璃  {self._side(ev['pan'])}  {tier}")
        elif ev["type"] == "action":
            self.logbox.insert(0, f"{stamp}  行为声(换弹/打药?)  {self._side(ev['pan'])}  {tier}")
        if self.logbox.size() > 40:
            self.logbox.delete(40)

    @staticmethod
    def _side(pan):
        if pan < -0.35:
            return "左"
        if pan > 0.35:
            return "右"
        return "中"

    def _redraw(self):
        snap = self.latest
        if snap is None:
            return
        self._draw_radar(snap)
        self._draw_bars(snap)
        self._draw_scope()

    def _draw_radar(self, snap):
        c = self.radar
        c.delete("all")
        cx, cy, r = 230, 400, 180
        # 扇形刻度 -90..90，正前朝上
        for deg in range(-90, 91, 15):
            a = math.radians(deg - 90)
            x1 = cx + (r - 14) * math.cos(a)
            y1 = cy + (r - 14) * math.sin(a)
            x2 = cx + r * math.cos(a)
            y2 = cy + r * math.sin(a)
            major = deg % 45 == 0
            c.create_line(x1, y1, x2, y2,
                          fill="#2a3546" if major else "#1c2430", width=2 if major else 1)
        c.create_arc(cx - r, cy - r, cx + r, cy + r, start=0, extent=180,
                     style="arc", outline="#2a3546", width=2)
        c.create_text(cx, cy + 20, text="前", fill="#6b7a8d",
                      font=("Microsoft YaHei UI", 10))
        c.create_text(cx - r - 14, cy, text="左", fill="#6b7a8d",
                      font=("Microsoft YaHei UI", 10))
        c.create_text(cx + r + 14, cy, text="右", fill="#6b7a8d",
                      font=("Microsoft YaHei UI", 10))

        if snap["present"] and snap["angle"] is not None:
            a = math.radians(snap["angle"] - 90)
            color = ACCENT if snap["tier"] == "近" else (WARN if snap["tier"] == "中" else FG)
            x = cx + (r - 30) * math.cos(a)
            y = cy + (r - 30) * math.sin(a)
            c.create_line(cx, cy, x, y, fill=color, width=3)
            rr = 10 if snap["tier"] == "近" else (7 if snap["tier"] == "中" else 5)
            c.create_oval(x - rr, y - rr, x + rr, y + rr, outline=color, width=2)
            c.create_text(cx, cy - 40, text=f"{snap['angle']:+.0f}°",
                          fill=color, font=("Consolas", 14, "bold"))
            self.status.config(text=f"真人脚步 · {snap['tier']}档", fg=color)
        else:
            self.status.config(text="无目标", fg="#5a6b7d")

        self.sub.config(text=f"低频底噪 {snap['floor']:.2e}    "
                             f"高频占比 {snap['feats']['high_ratio']:.2f}    "
                             f"掩蔽 {'开' if snap['masked'] else '关'}")

    def _draw_bars(self, snap):
        c = self.bars
        c.delete("all")
        f = snap["feats"]
        for i, (val, label) in enumerate(((f["rms_l"], "L"), (f["rms_r"], "R"))):
            y = 8 + i * 26
            w = min(1.0, val * 30) * 400
            color = ACCENT if (i == 0 and f["pan"] < -0.1) or (i == 1 and f["pan"] > 0.1) else "#3a4a5e"
            c.create_rectangle(30, y, 30 + w, y + 16, fill=color, outline="")
            c.create_text(16, y + 8, text=label, fill=FG, font=("Consolas", 10))

    def _draw_scope(self):
        c = self.scope
        c.delete("all")
        w = max(c.winfo_width(), 460)
        h = 120
        if len(self.hist) < 2:
            return
        lows = [x[0] for x in self.hist]
        peak = max(max(lows), 1e-7)
        for i in range(1, len(self.hist)):
            x1 = (i - 1) / 159 * w
            x2 = i / 159 * w
            low, floor = self.hist[i]
            y_low = h - 8 - min(1.0, low / peak) * (h - 16)
            y_fl = h - 8 - min(1.0, (floor or 0) / peak) * (h - 16)
            c.create_line(x1, y_fl, x2, y_fl, fill="#4a5568")
            c.create_line(x1, y_low, x2, y_low, fill=ACCENT, width=2)

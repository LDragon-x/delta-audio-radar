"""按键流记录：GetAsyncKeyState 物理直读，边沿检测写 keys.jsonl。

与音频/事件同一时钟（perf_counter - t0），以后和声音、画面对齐喂模型：
某个脚步/枪声响起时，用户手里按着什么键一目了然。

可选鼠标移动轨迹（mouse.jsonl）：同一轮询里 GetCursorPos 差分，
≥40ms 聚合一条 {"t","dx","dy"}——拉枪/转身/跟枪的动作流，
"听到声音后手是不是在动"也是好特征。

只记录边沿（按下/抬起）和移动差分，不轮询落盘；零注入零驱动（Win32 用户态 API）。
"""
import ctypes
import ctypes.wintypes
import json
import threading
import time

from dsp.selfsound import _VK


class KeyLogger:
    def __init__(self, path, t0_fn, keys, interval=0.02, mouse_path=None):
        """t0_fn: 返回会话时钟起点的 callable；keys: 要记录的键名列表。
        mouse_path: 给路径则同时记录鼠标移动轨迹（mouse.jsonl）。"""
        self.path = str(path)
        self.t0_fn = t0_fn
        self.keys = [k for k in dict.fromkeys(keys) if k in _VK]
        self.interval = interval
        self.edges = 0
        self.mouse_path = str(mouse_path) if mouse_path else None
        self.mouse_moves = 0
        self._stop = threading.Event()
        self._th = None
        self._err = None
        self._getasync = ctypes.windll.user32.GetAsyncKeyState
        self._getcursor = ctypes.windll.user32.GetCursorPos
        self._cursor = ctypes.wintypes.POINT()

    def start(self):
        if not self.keys and not self.mouse_path:
            self._err = "没有可记录的键（键名需在 VK 表里）"
            return
        self._th = threading.Thread(target=self._run, daemon=True,
                                    name="key-logger")
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=5)
        return self._err

    def _run(self):
        try:
            state = {k: False for k in self.keys}
            f = open(self.path, "w", encoding="utf-8")
            mf = open(self.mouse_path, "w", encoding="utf-8") \
                if self.mouse_path else None
            last_x = last_y = None
            acc_dx = acc_dy = 0.0
            last_write = -1.0
            try:
                while not self._stop.is_set():
                    t = time.perf_counter() - self.t0_fn()
                    for k in self.keys:
                        now = bool(self._getasync(_VK[k]) & 0x8000)
                        if now != state[k]:
                            state[k] = now
                            f.write(json.dumps({"t": round(t, 3), "key": k,
                                                "down": now},
                                               ensure_ascii=False) + "\n")
                            self.edges += 1
                    if mf:
                        # 鼠标移动差分：聚合 ≥40ms 写一条，静止零开销
                        if self._getcursor(ctypes.byref(self._cursor)):
                            cx, cy = self._cursor.x, self._cursor.y
                            if last_x is not None:
                                acc_dx += cx - last_x
                                acc_dy += cy - last_y
                            last_x, last_y = cx, cy
                            if (acc_dx or acc_dy) and t - last_write >= 0.04:
                                mf.write(json.dumps(
                                    {"t": round(t, 3),
                                     "dx": int(acc_dx),
                                     "dy": int(acc_dy)}) + "\n")
                                acc_dx = acc_dy = 0.0
                                last_write = t
                                self.mouse_moves += 1
                        mf.flush()
                    f.flush()
                    time.sleep(self.interval)
            finally:
                f.close()
                if mf:
                    mf.close()
        except Exception as e:  # 按键记录挂了不能拖垮采集主流程
            self._err = f"按键记录失败: {e}"

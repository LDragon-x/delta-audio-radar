"""屏幕录制：mss 截屏 + imageio-ffmpeg 编码 mp4，供标注时对照画面。

零注入零驱动：mss 是 ctypes 直读 GDI 位图，与 OBS 画布采集同级机制；
对局内只写视频文件，不读画面内容（不做任何像素分析）。

时钟与 LoopbackSource 同源（perf_counter - 音频 t0），标注页按事件 t
直接 seek 就能对齐到同一时刻的画面。

用法（采集器内部调用）：
    rec = ScreenRecorder("sessions/<ts>/video.mp4", anchor=lambda: src._t0)
    rec.start()
    ...
    rec.stop()   # finalize 里调，会等编码线程收尾
"""
import threading
import time

import numpy as np


class ScreenRecorder:
    def __init__(self, path, anchor, fps=10, width=1280):
        """anchor: 返回音频时钟 t0 的 callable（None 表示还没开始）。"""
        self.path = str(path)
        self.anchor = anchor
        self.fps = fps
        self.width = width
        self.frames = 0
        self._times = []  # 每帧的会话时间戳（t0 锚定），写 video_frames.json 供标注页精确对位
        self._stop = threading.Event()
        self._th = None
        self._err = None

    def start(self):
        self._th = threading.Thread(target=self._run, daemon=True,
                                    name="screen-recorder")
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=15)
        if self._times:
            try:
                import json as _json
                meta = {"fps": self.fps, "times": self._times}
                with open(self.path.replace(".mp4", "_frames.json"),
                          "w", encoding="utf-8") as f:
                    f.write(_json.dumps(meta))
            except Exception:
                pass
        return self._err

    def _run(self):
        try:
            import mss
            import imageio.v2 as imageio
            from PIL import Image

            with mss.mss() as sct:
                mon = sct.monitors[1]  # 主显示器
                raw_w, raw_h = mon["width"], mon["height"]
                scale = min(1.0, self.width / raw_w)
                w, h = int(raw_w * scale) & ~1, int(raw_h * scale) & ~1
                box = {"top": mon["top"], "left": mon["left"],
                       "width": raw_w, "height": raw_h}

                # 等音频时钟起锚，保证时间轴同源
                t0 = None
                while not self._stop.is_set():
                    t0 = self.anchor()
                    if t0 is not None:
                        break
                    time.sleep(0.05)
                if t0 is None:
                    return

                wri = imageio.get_writer(
                    self.path, fps=self.fps, codec="libx264",
                    quality=7, macro_block_size=2,
                    ffmpeg_params=["-preset", "veryfast", "-crf", "30",
                                   "-pix_fmt", "yuv420p"])
                try:
                    interval = 1.0 / self.fps
                    while not self._stop.is_set():
                        tic = time.perf_counter()
                        shot = sct.grab(box)
                        img = np.frombuffer(shot.raw, dtype=np.uint8) \
                            .reshape(shot.height, shot.width, 4)[..., :3]
                        if scale < 1.0:
                            img = np.asarray(Image.fromarray(img)
                                             .resize((w, h), Image.BILINEAR))
                        wri.append_data(img[:, :, ::-1])  # BGR->RGB
                        self.frames += 1
                        self._times.append(round(time.perf_counter() - t0, 3))
                        dt = time.perf_counter() - tic
                        if dt < interval:
                            time.sleep(interval - dt)
                finally:
                    wri.close()
        except Exception as e:  # 录屏挂了不能拖垮采集主流程
            self._err = f"录屏失败: {e}"

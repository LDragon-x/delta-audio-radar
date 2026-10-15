"""采集层：音频源抽象。

LoopbackSource  —— WASAPI loopback，旁路抓取送往指定输出设备（默认扬声器）的比特流。
                    游戏无感知，零注入，与 OBS 输出采集同级机制。
FileSource      —— 离线验证源：读 wav 文件按实时节奏吐块，模拟 loopback。
                    用三角洲对局录像提取的音频即可离线调参。

统一接口：sr 采样率，read() -> (t, float32 数组 [frames, 2]) 或 None 表示结束。
"""
import collections
import threading
import time

import numpy as np


class LoopbackSource:
    """WASAPI loopback（pyaudiowpatch 回调模式）。

    2026-10-01 实测：soundcard 的 loopback 实现（ctypes MediaFoundation）
    在本机跨进程捕获必崩（同进程放音可录、其他进程放音返回空/纯零），
    pyaudiowpatch 回调模式跨进程回环正常。阻塞式 read() 会挂死，
    必须 stream_callback + 缓冲队列。
    """

    def __init__(self, device_keyword=None, sr=48000, block=2048):
        import pyaudiowpatch as pyaudio
        self.sr = sr
        self.block = block
        pa = pyaudio.PyAudio()
        self._pa = pa
        devs = []
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if d.get("maxInputChannels", 0) > 0:
                devs.append(d)
        loopbacks = [d for d in devs if "loopback" in d["name"].lower()]
        if not loopbacks:
            pa.terminate()
            raise RuntimeError("没有找到 loopback 设备（需要 Windows + 音频输出设备）")
        pick = None
        if device_keyword:
            for d in loopbacks:
                if device_keyword in d["name"]:
                    pick = d
                    break
        if pick is None:
            # 2026-10-01 实测：本机 USB 耳机（GAMING/CHAT）的 WASAPI loopback
            # 端点不送数据（回调零触发），VB 虚拟线（入梦扬声器）正常。
            # 默认优先虚拟线，耳机 loopback 需显式 keyword 指定。
            for d in loopbacks:
                nm = d["name"]
                if "VB" in nm or "Cable" in nm or "入梦" in nm:
                    pick = d
                    break
        if pick is None:
            pick = loopbacks[0]
        self.device_name = pick["name"]
        # 用设备原生采样率打开，read() 里重采样到目标 sr
        dev_sr = int(pick.get("defaultSampleRate") or sr)
        self._dev_sr = dev_sr
        self._q = collections.deque(maxlen=4096)
        self._cond = threading.Condition()

        def _cb(in_data, frame_count, time_info, status):
            with self._cond:
                self._q.append(np.frombuffer(in_data, dtype=np.float32).copy())
                self._cond.notify()
            return (None, pyaudio.paContinue)

        try:
            self._stream = pa.open(
                format=pyaudio.paFloat32, channels=2, rate=dev_sr,
                input=True, input_device_index=int(pick["index"]),
                frames_per_buffer=block, stream_callback=_cb)
            self._stream.start_stream()
        except Exception:
            pa.terminate()
            raise
        self._t0 = None

    def _pop(self):
        with self._cond:
            while not self._q:
                self._cond.wait(timeout=0.5)
                if not self._stream.is_active():
                    return None
            return self._q.popleft()

    def read(self):
        """攒满 block 帧再返回 (t, (block, 2) float32)；设备流断开返回 None。"""
        chunks = []
        got = 0
        while got < self.block:
            c = self._pop()
            if c is None:
                return None
            chunks.append(c)
            got += len(c) // 2
        raw = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        frames = raw[: self.block * 2].reshape(-1, 2)
        if len(frames) > self.block:  # 多拿的塞回队首
            extra = frames[self.block:].reshape(-1)
            with self._cond:
                self._q.appendleft(extra)
        if self._dev_sr != self.sr:  # 重采样到目标 sr
            n = len(frames)
            x_old = np.arange(n) / self._dev_sr
            x_new = np.arange(int(n * self.sr / self._dev_sr)) / self.sr
            frames = np.stack(
                [np.interp(x_new, x_old, frames[:, ch]) for ch in range(2)],
                axis=1).astype(np.float32)
        if self._t0 is None:
            self._t0 = time.perf_counter()
        t = time.perf_counter() - self._t0
        return t, np.asarray(frames, dtype=np.float32)

    def close(self):
        try:
            self._stream.stop_stream()
            self._stream.close()
        except Exception:
            pass
        try:
            self._pa.terminate()
        except Exception:
            pass


class FileSource:
    """wav 文件按 2 倍速节奏回放（调参时不用等全程）。"""

    def __init__(self, path, sr=48000, block=2048, speed=2.0, skip=0.0):
        import wave
        w = wave.open(path, "rb")
        self.sr = w.getframerate()
        ch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        raw = raw.reshape(-1, ch)
        if ch == 1:
            raw = np.repeat(raw, 2, axis=1)
        self.data = raw[:, :2]
        self.block = block
        self.speed = speed
        self.pos = int(skip * self.sr)
        self._t0 = None
        w.close()

    def read(self):
        if self.pos + self.block > len(self.data):
            return None
        chunk = self.data[self.pos:self.pos + self.block]
        self.pos += self.block
        t = self.pos / self.sr
        time.sleep(self.block / self.sr / self.speed)
        return t, chunk

    def close(self):
        pass  # 无外部资源；worker 统一调 close()，不能让它 AttributeError


def list_loopback_devices():
    import pyaudiowpatch as pyaudio
    pa = pyaudio.PyAudio()
    names = []
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d.get("maxInputChannels", 0) > 0 and "loopback" in d["name"].lower():
            names.append(d["name"])
    pa.terminate()
    return names

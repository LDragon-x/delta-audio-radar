"""实时多频段音效处理器：缩小枪声 / 抬脚步 / 压环境杂音。

设计目标（对应玩家刚需"缩小枪声放大脚步"）：
- 低频段 (<250Hz)：恒定提升（脚步躯干）
- 中频段 (250-2kHz)：直通（人声/报点不动）
- 高频段 (>2kHz)：向下压缩——慢平均跟踪底噪，突发 surge（枪声/爆炸/碎玻璃）
  快速压增益，最多压到 high_floor；释放慢，避免底噪"pumping"

分带：2048 点矩形分析 FFT（频带很宽，泄漏可忽略）+ Hann 合成窗重叠相加(OLA)，
hop 1024。COLA 恒等性由合成窗保证，处理增益缓变时无块间痕迹。
理论延迟 ≈ block/sr ≈ 43ms + 声卡双向缓冲（实测 typically 80-120ms）。
"""
import numpy as np

SR = 48000
BLOCK = 2048
HOP = 1024
EDGES = [(0, 250), (250, 2000), (2000, 24000)]


class MultibandFX:
    def __init__(self, low_boost_db=5.0, high_floor_gain=0.25,
                 surge_ratio=4.0, attack=0.35, release=0.08):
        self.low_gain = 10 ** (low_boost_db / 20)
        self.high_floor = high_floor_gain
        self.surge_ratio = surge_ratio
        self.attack = attack      # 朝压缩目标靠近的速度（快）
        self.release = release    # 增益恢复的速度（慢，防起伏）
        self.win = np.hanning(BLOCK)
        self.buf = np.zeros((0, 2), dtype=np.float32)   # 待分帧输入
        self.acc = np.zeros((BLOCK, 2), dtype=np.float32)  # OLA 累加器
        self.slow_high = None
        self.gain_high = 1.0
        self._masks = None

    def _band_masks(self):
        if self._masks is None:
            fr = np.fft.rfftfreq(BLOCK, 1 / SR)
            self._masks = [(fr >= lo) & (fr < hi) for lo, hi in EDGES]
        return self._masks

    def _process_block(self, frame):
        """frame: [BLOCK, 2] -> [BLOCK, 2]（未加合成窗）"""
        masks = self._band_masks()
        gains = np.array([self.low_gain, 1.0, self.gain_high])
        out = np.zeros_like(frame)
        spec = [np.fft.rfft(frame[:, ch]) for ch in range(2)]
        for mi, mask in enumerate(masks):
            g = gains[mi]
            for ch in range(2):
                spec[ch][mask] *= g
        for ch in range(2):
            out[:, ch] = np.fft.irfft(spec[ch], BLOCK)
        return out

    def _update_envelope(self, frame):
        """高频压缩包络：慢平均跟踪底噪，surge 快速压增益。"""
        mono = frame.mean(axis=1)
        high_pow = float(np.sum(np.abs(np.fft.rfft(mono))[self._band_masks()[2]]) ** 2)
        if self.slow_high is None:
            self.slow_high = high_pow + 1e-12
        target = 1.0
        if high_pow > self.slow_high * self.surge_ratio:
            target = max(self.high_floor, (self.slow_high * self.surge_ratio) / high_pow)
        rate = self.attack if target < self.gain_high else self.release
        self.gain_high += (target - self.gain_high) * rate
        if high_pow < self.slow_high * self.surge_ratio:
            self.slow_high = 0.98 * self.slow_high + 0.02 * high_pow

    def process(self, chunk):
        """chunk: [n, 2] 任意长度，返回处理后的音频流（长度≈n-BLOCK，含分帧延迟）。"""
        self.buf = np.concatenate([self.buf, np.asarray(chunk, dtype=np.float32)], axis=0)
        emitted = []
        while len(self.buf) >= BLOCK:
            frame = self.buf[:BLOCK]
            self.buf = self.buf[HOP:]
            self._update_envelope(frame)
            y = self._process_block(frame) * self.win[:, None]
            emitted.append(self.acc[:HOP] + y[:HOP])
            self.acc = np.concatenate([self.acc[HOP:], y[HOP:]], axis=0)
        if not emitted:
            return np.zeros((0, 2), dtype=np.float32)
        result = np.concatenate(emitted, axis=0)
        peak = float(np.max(np.abs(result))) + 1e-9
        if peak > 0.98:
            result = result * (0.98 / peak)
        return result.astype(np.float32)

    def flush(self):
        """尾部残余（约 BLOCK 样本）。"""
        tail = self.acc[:HOP].copy()
        self.acc[:] = 0
        self.buf = np.zeros((0, 2), dtype=np.float32)
        return tail


def render_file(in_path, out_path, sr=SR, **fx_kw):
    """离线渲染：整文件过一遍 FX，供 A/B 试听。"""
    import wave
    w = wave.open(str(in_path), "rb")
    src_sr = w.getframerate()
    raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    data = raw.reshape(-1, w.getnchannels())
    w.close()
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    if src_sr != sr:
        raise ValueError(f"需要 {sr}Hz 音频，实际 {src_sr}Hz")
    fx = MultibandFX(**fx_kw)
    out = []
    step = HOP * 8
    for i in range(0, len(data) - BLOCK, step):
        out.append(fx.process(data[i:i + step]))
    out.append(fx.flush())
    result = np.concatenate([o for o in out if len(o)], axis=0)
    result = result[:len(data)]
    if len(result) < len(data):
        result = np.pad(result, ((0, len(data) - len(result)), (0, 0)))
    pcm = (np.clip(result, -1, 1) * 32767).astype(np.int16)
    wo = wave.open(str(out_path), "wb")
    wo.setnchannels(2)
    wo.setsampwidth(2)
    wo.setframerate(sr)
    wo.writeframes(pcm.tobytes())
    wo.close()
    return out_path

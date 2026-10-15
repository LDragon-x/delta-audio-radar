"""帧级自我消噪模型 v0.1（DeepFilterNet/RNNoise 路线的小号本地版）。

与事件门控（rt_suppress.py）的本质区别：
  - 输出连续掩码（每 8ms 帧 × 每 mel 频带一个增益），不是事件窗整段压；
  - 因果推理：只看过去 CTX 帧，决策延迟 = 单帧 8ms + 上下文 200ms；
  - 训练用 mix-and-mask：靶场纯享自己脚步随机混入实战音频，
    逐时频格算"自己分量"的 Wiener 增益当真值，帧级监督无死角。

信号流（16k mono，与全管线一致）：
  STFT(512/128) → 功率 → mel 64 带 → log → GRU → sigmoid 增益(0..1)
  → 增益插值回 STFT 频带 → ISTFT 重叠相加出波形。

2026-10-01：v0.1 骨架，训练脚本 tool/train_frame.py。
"""
import numpy as np
import torch
import torch.nn as nn

SR = 16000
N_FFT = 512
HOP = 128          # 8ms
N_MELS = 64
FMIN = 40.0
CTX = 25           # 因果上下文帧数（25×8ms = 200ms）
WIN = np.hanning(N_FFT).astype(np.float32)
# hann 512 / hop 128 满足 COLA，重叠相加直接归一


def stft(x):
    """(n,) → 功率谱 (frames, n_fft//2+1)。与 dsp.logmel.stft_power 同参。"""
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n_fr = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_fr)[:, None]
    frames = x[idx] * WIN
    spec = np.fft.rfft(frames, axis=1)
    return (spec.real ** 2 + spec.imag ** 2).astype(np.float32), spec


def istft(spec):
    """复数谱 (frames, n_fft//2+1) → 波形（COLA 归一）。"""
    n_fr = spec.shape[0]
    out = np.zeros(N_FFT + HOP * (n_fr - 1), dtype=np.float32)
    win_acc = np.zeros_like(out)
    frames = np.fft.irfft(spec, n=N_FFT, axis=1)
    for i in range(n_fr):
        s = i * HOP
        out[s:s + N_FFT] += frames[i] * WIN
        win_acc[s:s + N_FFT] += WIN ** 2
    return out / np.maximum(win_acc, 1e-8)


_mel = None


def mel_fb():
    global _mel
    if _mel is None:
        from dsp.logmel import mel_bank
        _mel = mel_bank(SR, N_FFT, N_MELS, FMIN, None).astype(np.float32)
    return _mel


def feats_pow(power):
    """STFT 功率 (fr, bin) → log-mel (fr, N_MELS)。"""
    m = power @ mel_fb().T
    return np.log(m + 1e-10)


def target_gain(p_bg, p_self):
    """Wiener 增益：保留背景、抑制自己。mel 带口径，(fr, N_MELS) ∈ [0,1]。"""
    mb = p_bg @ mel_fb().T
    ms = p_self @ mel_fb().T
    return (mb / (mb + ms + 1e-12)).astype(np.float32)


class FrameNet(nn.Module):
    """log-mel(64) 帧序列 → 每帧每带增益。因果：逐帧吃，内部状态滚动。"""

    def __init__(self, n_mels=N_MELS, hid=128, layers=2):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(n_mels), nn.Linear(n_mels, hid),
                                 nn.ReLU())
        self.gru = nn.GRU(hid, hid, layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, n_mels), nn.Sigmoid())

    def forward(self, x):
        """x: (B, T, n_mels) → g: (B, T, n_mels)"""
        z, _ = self.gru(self.inp(x))
        return self.head(z)


class GatedBlock(nn.Module):
    """WaveNet 门控因果卷积残差块：tanh(Wf*x) * sigmoid(Wg*x)。"""

    def __init__(self, ch, dil, k=3):
        super().__init__()
        self.pad = (k - 1) * dil
        self.f = nn.Conv1d(ch, ch, k, dilation=dil)
        self.g = nn.Conv1d(ch, ch, k, dilation=dil)
        self.res = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        xp = torch.nn.functional.pad(x, (self.pad, 0))
        h = torch.tanh(self.f(xp)) * torch.sigmoid(self.g(xp))
        return x + self.res(h)


class FrameNorm(nn.Module):
    """nn.LayerNorm 的手工等价（逐帧 64 维标准化+仿射）。

    DirectML 反向 bug 定位（2026-10-01）：nn.LayerNorm 与后续卷积块
    组合时 CompileGraph 必失败，单算子均正常。拆成 mean/var 基础算子
    后绕过。数值与 nn.LayerNorm(eps=1e-5, elementwise_affine=True) 等价。
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = (x - mu).square().mean(dim=-1, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias


class FrameNet2(nn.Module):
    """v0.4 因果膨胀卷积版（WaveNet 式，GRU 的 DirectML 替代）。

    与 FrameNet 同为因果（输出 t 只看 ≤t），但卷积可并行、全算子 DML 兼容，
    训练在 AMD 显卡上跑。支持条件向量（姿态/地形 one-hot 或嵌入），
    FiLM 注入输入层；cond=0 时退化为无条件模型。
    """

    def __init__(self, n_mels=N_MELS, hid=96, layers=6, k=3, cond=0):
        super().__init__()
        self.n_mels = n_mels
        self.cond_dim = cond
        self.inp = nn.Sequential(FrameNorm(n_mels), nn.Linear(n_mels, hid),
                                 nn.ReLU())
        if cond:
            self.film = nn.Linear(cond, 2 * hid)   # 输入 FiLM（scale/shift）
            nn.init.zeros_(self.film.bias)
            nn.init.zeros_(self.film.weight)
        else:
            self.film = None
        self.blocks = nn.ModuleList(
            [GatedBlock(hid, dil=min(2 ** i, 32), k=k) for i in range(layers)])
        self.head = nn.Sequential(nn.Linear(hid, n_mels), nn.Sigmoid())
        self.rf = sum((k - 1) * min(2 ** i, 32) for i in range(layers)) + 1

    def forward(self, x, cond=None):
        """x: (B, T, n_mels)，cond: (cond_dim,)/(B, cond_dim)/(B, T, cond_dim)
        或 None → g: (B, T, n_mels)。per-frame 条件时 FiLM 逐帧生效。"""
        z = self.inp(x)
        if self.film is not None:
            c = cond if cond is not None else torch.zeros(
                x.shape[0], self.cond_dim, device=x.device)
            if c.dim() == 1:
                c = c.view(1, 1, -1)             # (cond,) 单条件
            elif c.dim() == 2:
                c = c[:, None, :]                # (B, cond) → 全帧同条件
            g, b = self.film(c).chunk(2, dim=-1)
            z = z * (1.0 + g) + b
        h = z.transpose(1, 2)              # (B, hid, T)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h.transpose(1, 2))



class FrameSuppressor:
    """流式推理封装：逐块喂波形，出消噪波形。延迟 = CTX 帧（200ms）。"""

    def __init__(self, ckpt_path, device="cpu"):
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        self.mu = torch.tensor(ck["mu"], dtype=torch.float32, device=device)
        self.sd = torch.tensor(ck["sd"], dtype=torch.float32, device=device)
        arch = dict(ck.get("arch", {}))
        self.arch_type = arch.pop("type", "gru")
        if self.arch_type == "tcn2":
            self.net = FrameNet2(**arch).to(device)
        else:
            self.net = FrameNet(**arch).to(device)
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.device = device
        self.h = None          # GRU 隐状态
        self.feat_buf = None   # log-mel 上下文缓冲
        self.cond = None       # (1, cond_dim) 条件向量（姿态/地形）
        self.rf = getattr(self.net, "rf", 0)   # TCN 感受野（帧）

    def set_condition(self, vec):
        """设条件向量（姿态/地形 one-hot），形状 (cond_dim,) 或 None。"""
        if vec is None:
            self.cond = None
        else:
            import numpy as _np
            v = _np.asarray(vec, dtype=_np.float32).reshape(1, -1)
            self.cond = torch.tensor(v, device=self.device)

    def reset(self):
        self.h = None
        self.feat_buf = None

    @torch.no_grad()
    def process_block(self, x):
        """x: (n,) numpy 16k → 同长消噪波形。块长任意（内部按帧滚动）。

        GRU 前向是因果的（t 时刻输出只依赖 ≤t 输入），整段跑即流式，
        无需逐帧手动喂——只在块边界保留 CTX 帧特征与隐状态。
        TCN 同理因果，保留 rf 帧输入特征缓冲即可。
        尾部不足一整帧的样本直通不处理。"""
        n_full = ((len(x) - N_FFT) // HOP) * HOP + N_FFT
        if n_full < N_FFT:
            return x.copy()
        tail = x[n_full:]
        x = x[:n_full]
        power, spec = stft(x)
        lm = feats_pow(power)                     # (fr, M)
        t = torch.tensor(lm, dtype=torch.float32,
                         device=self.device)[None]
        t = ((t - self.mu) / self.sd).clamp(-5.0, 5.0)   # v0.3 与训练同口径截断
        if self.arch_type == "tcn2":
            # 保留 rf 帧历史，输出仅取本块对应帧（因果裁剪）
            if self.feat_buf is not None:
                t = torch.cat([self.feat_buf, t], dim=1)
            need = max(0, self.rf - t.shape[1])   # 仅当总长度不足感受野才补零
            if need > 0:
                t = torch.nn.functional.pad(t, (0, 0, need, 0))
            self.feat_buf = t[:, -(self.rf - 1):].detach().clone() \
                if self.rf > 1 else None
            g = self.net(t, self.cond)
            g_np = g[0, -len(power):].cpu().numpy().astype(np.float32)
        else:
            if self.feat_buf is not None:
                t = torch.cat([self.feat_buf, t], dim=1)
            else:
                t = torch.nn.functional.pad(t, (0, 0, CTX, 0))
            z, self.h = self.net.gru(self.net.inp(t), self.h)
            g = self.net.head(z)
            g_np = g[0].cpu().numpy().astype(np.float32)   # (T, M)
            self.feat_buf = t[:, -CTX:].detach().clone()
        gb = mel_to_bins(g_np)                     # 只取本块帧
        return np.concatenate([istft(spec * np.sqrt(gb)), tail])


def mel_to_bins(g_mel):
    """(fr, N_MELS) → (fr, n_bins)：mel 带边界频率处线性插值。"""
    from dsp.logmel import mel_to_hz, hz_to_mel
    n_bins = N_FFT // 2 + 1
    freqs = np.linspace(0, SR / 2, n_bins)
    # 各 mel 带中心频率（mel 滤波行均值位置近似取带中心）
    edges = np.linspace(hz_to_mel(FMIN), hz_to_mel(SR / 2), N_MELS + 1)
    centers = mel_to_hz((edges[:-1] + edges[1:]) / 2)
    centers = np.concatenate([[0.0], centers, [SR / 2]])
    gm = np.concatenate([g_mel[:, :1], g_mel, g_mel[:, -1:]], axis=1)
    return np.stack([np.interp(freqs, centers, gm[i]) for i in range(g_mel.shape[0])]).astype(np.float32)

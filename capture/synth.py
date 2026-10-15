"""合成测试源：生成带脚/枪声的双声道测试信号，离线验证整条 DSP 链路。

模拟场景（10 秒，48kHz 立体声）：
  1.0s  脚步 偏左 近
  2.0s  脚步 正中 中
  3.2s  脚步 偏右 近
  4.5s  密集环境噪声抬升（模拟嘈杂战场）
  5.0s  脚步 偏左 近（噪声中，考验相对底噪触发）
  6.0s  枪声 偏右
  6.3s  脚步（掩蔽窗口内，应被丢弃）
  8.0s  脚步 偏右 远
"""
import numpy as np

SR = 48000
DUR = 11.5
# 共享随机源：每次调用工厂都产生独立噪声 → 左右耳去相关
_RNG = np.random.default_rng(7)


def _thump(n, bright):
    """单声脚步：低通噪声脉冲，bright 越高越亮（越近）。

    one-pole 低通系数压得很低，保证低频主导——真实脚步 80-500Hz
    占绝对大头，之前的系数导致中频喧宾夺主。
    """
    noise = _RNG.standard_normal(n)
    alpha = 0.008 + 0.05 * bright
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc += alpha * (noise[i] - acc)
        y[i] = acc
    env = np.exp(-np.linspace(0, 5, n))
    return y * env


def _gunshot(n):
    return _RNG.standard_normal(n) * np.exp(-np.linspace(0, 8, n))


def _glass(n):
    """碎玻璃：高通噪声快衰减（一阶差分近似高通），高频主导。"""
    noise = _RNG.standard_normal(n)
    hp = np.diff(noise, prepend=0.0)
    return hp * np.exp(-np.linspace(0, 10, n))


def _place(make_sig, total, start, pan, gain):
    """放置一个事件。make_sig 是信号工厂，左右耳各调一次——
    真实 HRTF 下双耳信号是部分去相关的，demo 也必须如此，否则会被单声道门限误杀。"""
    n = len(make_sig())
    s = int(start * SR)
    if s + n > total:
        n = total - s
    gl = gain * min(1.0, np.sqrt((1 - pan) / 2) * np.sqrt(2))
    gr = gain * min(1.0, np.sqrt((1 + pan) / 2) * np.sqrt(2))
    out_l[s:s + n] += make_sig()[:n] * gl
    out_r[s:s + n] += make_sig()[:n] * gr


def _place_mono(make_sig, total, start, gain):
    """自听音效：左右耳同一信号（近单声道）且全场最响——自己的脚步/动作。"""
    n = len(make_sig())
    s = int(start * SR)
    if s + n > total:
        n = total - s
    sig = make_sig()[:n] * gain
    out_l[s:s + n] += sig
    out_r[s:s + n] += sig


total = int(DUR * SR)
out_l = np.zeros(total)
out_r = np.zeros(total)
rng = np.random.default_rng(3)

# 环境底噪（全程轻微）
amb = rng.standard_normal(total) * 0.002
out_l += amb
out_r += rng.standard_normal(total) * 0.002

# 4.5s 起战场嘈杂：低频轰鸣（左右独立 + 淡入，模拟 HRTF 环境声）+ 随机低频闷响
t0 = int(4.5 * SR)
k = np.ones(400) / 400
rumble_l = np.convolve(rng.standard_normal(total - t0) * 0.010, k, mode="same")
rumble_r = np.convolve(rng.standard_normal(total - t0) * 0.010, k, mode="same")
fade = np.linspace(0, 1, int(0.1 * SR))  # 100ms 淡入，真实噪声不会阶跃出现
rumble_l[:len(fade)] *= fade
rumble_r[:len(fade)] *= fade
out_l[t0:] += rumble_l
out_r[t0:] += rumble_r
for _ in range(12):
    st = t0 + rng.integers(0, total - t0 - 2000)
    # 闷响：白噪声过低通，左右独立（HRTF 环境声不是单声道拷贝）
    blip = np.convolve(rng.standard_normal(1500) * 0.01, k, mode="same") \
        * np.exp(-np.linspace(0, 4, 1500))
    blip2 = np.convolve(rng.standard_normal(1500) * 0.01, k, mode="same") \
        * np.exp(-np.linspace(0, 4, 1500))
    out_l[st:st + 1500] += blip
    out_r[st:st + 1500] += blip2

N = int(0.12 * SR)
_place(lambda: _thump(N, 0.8), total, 1.0, -0.6, 0.9)   # 偏左 近
_place(lambda: _thump(N, 0.5), total, 2.0, 0.0, 0.5)    # 正中 中
_place(lambda: _thump(N, 0.8), total, 3.2, 0.6, 0.9)    # 偏右 近
_place(lambda: _thump(N, 0.8), total, 5.0, -0.6, 0.9)   # 嘈杂中 偏左 近
_place(lambda: _gunshot(int(0.3 * SR)), total, 6.0, 0.5, 1.2)  # 枪声
_place(lambda: _thump(N, 0.9), total, 6.3, -0.3, 1.0)   # 掩蔽窗口内，应丢弃
_place(lambda: _thump(N, 0.3), total, 8.0, 0.7, 0.25)   # 偏右 远
_place(lambda: _thump(N, 0.1), total, 9.2, -0.4, 0.12)  # 6米级静步，静步档应捕获
_place(lambda: _glass(int(0.25 * SR)), total, 9.8, 0.3, 0.8)  # 碎玻璃
_place_mono(lambda: _thump(N, 0.6), total, 10.4, 1.0)          # 自己的脚步：最响+居中+单声道

demo_audio = np.stack([out_l, out_r], axis=1).astype(np.float32)


def demo_blocks(block=2048):
    """按块切分合成音频（供离线验证，不含 sleep）。"""
    for i in range(0, len(demo_audio) - block, block):
        t = i / SR
        yield t, demo_audio[i:i + block]

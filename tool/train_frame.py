"""帧级消噪模型训练 v0.1：mix-and-mask 监督（2026-10-01）。

监督构造：
  自己源 = 靶场纯享会话（默认 20261001_150719，全是自己脚步，键盘背书）
  背景源 = 实战会话音频（self 事件 ±1.2s 内的段剔除，剩下的当"别人/环境"）
  混合   y = bg + α·self，α ~ logU(-20dB, -4dB)（相对背景 RMS）
  真值   G = P_bg/(P_bg+P_self)  per mel 带（Wiener 增益，保背景抑自己）

训练：4s 窗、batch 32、Adam+MSE（对 G<0.9 的"该压"区加 2 倍权重）。
评测：同 self_suppress 口径——跑整局 125839，算 self 事件深压率/中位 dB
与 other 事件深误伤率，和事件门控现网指标（53.1% / 17.4%）直接对比。

用法：
  python tool/train_frame.py                      # 默认全流程
  python tool/train_frame.py --epochs 30 --no-train   # 只评测已有 ckpt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tool"))

import learn_embed as le  # noqa: E402  (load_mono, TARGET_SR)
import pose_label as pl  # noqa: E402  (pose_timeline, pose_at)
from dsp.frame_model import (CTX, HOP, N_FFT, SR, FrameNet, FrameNet2,  # noqa: E402
                             feats_pow, mel_fb, stft, target_gain)

SELF_SESSIONS = ["20261001_150719", "20261001_150622", "20261001_164045"]
BG_SESSIONS = ["20261001_125839", "20260930_211524",
               "20260927_185041", "20260927_164645"]
G_MIN = 10.0 ** (-25.0 / 20.0)     # 与线上一致：深压判定阈值
POSES_ALL = ["none", "walk", "run", "jump", "crouch", "crawl"]  # 条件 one-hot 词表
_pose_tls = {}                       # sid -> 姿态时间线（惰性加载）


def pose_tl(sid):
    if sid not in _pose_tls:
        kp = ROOT / "sessions" / sid / "keys.jsonl"
        _pose_tls[sid] = pl.pose_timeline(kp) if kp.exists() else []
    return _pose_tls[sid]
SELF_EXCLUDE = 1.2                 # self 事件前后排除秒数


def load_bg_pool(sess_ids):
    """背景池：(audio, sr) 列表 + 每段的 self 排除区间已抠掉。"""
    pool = []
    for sid in sess_ids:
        d = ROOT / "sessions" / sid
        if not (d / "auto_events.jsonl").exists():
            continue
        mono = le.load_mono(d)
        ev = [json.loads(l) for l in open(d / "auto_events.jsonl", encoding="utf-8")]
        bad = np.zeros(len(mono), dtype=bool)
        for e in ev:
            if e.get("self"):
                c = int(e["t"] * le.TARGET_SR)
                bad[max(0, c - int(SELF_EXCLUDE * le.TARGET_SR)):
                    min(len(mono), c + int(SELF_EXCLUDE * le.TARGET_SR))] = True
        good = ~bad
        # 连续 True 段切成样本
        idx = np.flatnonzero(np.diff(np.concatenate([[0], good.view(np.int8), [0]])))
        for a, b in idx.reshape(-1, 2):
            if b - a > le.TARGET_SR:
                pool.append(mono[a:b])
        print(f"  背景 {sid}: {len(pool)} 段可用（剔 self 后）")
    return pool


def mix_one(bg_pool, self_pool, rng, dur=4.0):
    """造一个混合样本 → (y, G_target, cond)。返回 torch 张量。"""
    n = int(dur * SR)
    bg = np.zeros(n, dtype=np.float32)
    # 背景拼接 1~2 段
    i = 0
    while i < n:
        seg = bg_pool[rng.integers(len(bg_pool))]
        take = min(n - i, len(seg))
        s0 = rng.integers(0, max(1, len(seg) - take))
        bg[i:i + take] = seg[s0:s0 + take]
        i += take
    self_sid, sl = self_pool[rng.integers(len(self_pool))]
    s0 = rng.integers(0, max(1, len(sl) - n))
    self_sig = np.zeros(n, dtype=np.float32)
    self_sig[:min(n, len(sl) - s0)] = sl[s0:s0 + n]
    # 随机 α：logU(-18dB, 0dB)（v0.3 加宽——实战自己脚步常常不弱于背景，
    # v0.2 只见过 -15..-3dB 的弱自噪，对响自己 OOD）
    ratio = 10.0 ** (rng.uniform(-18.0, 0.0) / 20.0)
    rms_bg = np.sqrt(np.mean(bg ** 2)) + 1e-8
    rms_sl = np.sqrt(np.mean(self_sig ** 2)) + 1e-8
    self_sig *= ratio * rms_bg / rms_sl
    y = bg + self_sig
    # 真值增益
    p_bg, _ = stft(bg)
    p_sl, _ = stft(self_sig)
    G = target_gain(p_bg, p_sl)
    lm = feats_pow(stft(y)[0])
    n_fr = lm.shape[0]
    cond = _frame_pose_cond(n_fr, self_sid, s0)
    return (torch.tensor(lm, dtype=torch.float32),
            torch.tensor(G, dtype=torch.float32), cond)


def _frame_pose_cond(n_fr, sid, s0):
    """帧级姿态 one-hot：(n_fr, 6)。self 帧按时间线查姿态，无时间线全 none。
    帧中心时刻 = (s0 + j*HOP + N_FFT/2) / 16000（self 源为 16k 单声道）。"""
    cond = np.zeros((n_fr, len(POSES_ALL)), dtype=np.float32)
    cond[:, 0] = 1.0                     # 默认 none
    tl = pose_tl(sid)
    if not tl:
        return torch.tensor(cond)
    for j in range(n_fr):
        t = (s0 + j * HOP + N_FFT // 2) / SR
        pse = pl.pose_at(tl, t)
        if pse in POSES_ALL[1:]:
            cond[j] = 0.0
            cond[j, POSES_ALL.index(pse)] = 1.0
    return torch.tensor(cond)


def pure_bg_sample(bg_pool, rng, dur=4.0):
    """v0.3：30% 样本为纯背景（G≈1）。实战大部分时间是"没有自己脚步"，
    v0.2 训练集每个样本都混了自噪，模型从没见过纯背景分布——
    实测已训模型在真实音频上输出塌缩为常数 0.502（GRU 驻点）。"""
    n = int(dur * SR)
    bg = np.zeros(n, dtype=np.float32)
    i = 0
    while i < n:
        seg = bg_pool[rng.integers(len(bg_pool))]
        take = min(n - i, len(seg))
        s0 = rng.integers(0, max(1, len(seg) - take))
        bg[i:i + take] = seg[s0:s0 + take]
        i += take
    p_bg, _ = stft(bg)
    lm = feats_pow(p_bg)
    G = np.ones_like(lm)
    cond = np.zeros((lm.shape[0], len(POSES_ALL)), dtype=np.float32)
    cond[:, 0] = 1.0
    return (torch.tensor(lm, dtype=torch.float32),
            torch.tensor(G, dtype=torch.float32),
            torch.tensor(cond))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-mix", type=int, default=20000, help="混合样本总量")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--ckpt", default=str(ROOT / "dsp" / "models" / "frame_v0.pt"))
    ap.add_argument("--eval-session", default="20261001_125839")
    ap.add_argument("--arch", choices=["gru", "tcn2"], default="gru",
                    help="backbone：gru=原版（CPU），tcn2=因果膨胀卷积（DirectML 兼容）")
    ap.add_argument("--device", default="cpu", choices=["cpu", "directml"],
                    help="训练设备；directml 用 AMD 显卡（仅 tcn2）")
    ap.add_argument("--hid", type=int, default=96, help="tcn2 隐藏通道")
    ap.add_argument("--layers", type=int, default=6, help="tcn2 块数")
    ap.add_argument("--cond", type=int, default=0,
                    help="条件向量维度（0=无条件；6=姿态 one-hot none/walk/run/jump/crouch/crawl）")
    args = ap.parse_args()
    import os
    torch.set_num_threads(int(os.environ.get("FRAME_THREADS", "0")) or (os.cpu_count() or 8))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ckpt = Path(args.ckpt)
    if not args.no_train:
        print("== 加载素材 ==")
        self_pool = []
        for sid in SELF_SESSIONS:
            d = ROOT / "sessions" / sid
            if (d / "labeling").exists() or (d / "session.wav").exists():
                self_pool.append((sid, le.load_mono(d)))
                print(f"  自己源 {sid}: {len(self_pool[-1][1]) / SR:.0f}s")
        bg_pool = load_bg_pool(BG_SESSIONS)
        if not self_pool or not bg_pool:
            print("素材不足，退出")
            return

        print("== 预生成混合样本 ==", flush=True)
        t0 = time.time()
        X, Y, C = [], [], []
        for i in range(args.n_mix):
            if rng.random() < 0.3:   # v0.3：纯背景样本（G≈1），治真实音频驻点塌缩
                x, g, c = pure_bg_sample(bg_pool, rng)
            else:
                x, g, c = mix_one(bg_pool, self_pool, rng)
            X.append(x.half())                   # fp16 存特征，省一半内存
            Y.append(g.half())
            C.append(c.half())
            if (i + 1) % 4000 == 0:
                print(f"  {i + 1}/{args.n_mix} ({time.time() - t0:.0f}s)", flush=True)
        # mu/sd 用前 4000 个样本估计（log-mel 统计平稳，子样足够代表）
        head = torch.cat(X[:4000]).float()
        mu = head.mean(dim=0)                  # v0.3：逐频带（v0.2 误存标量，
        sd = head.std(dim=0).clamp_min(1e-5)   # 归一失真助长了驻点塌缩）
        ds = torch.utils.data.TensorDataset(torch.stack(X), torch.stack(Y),
                                            torch.stack(C))
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch,
                                         shuffle=True, drop_last=True)
        print(f"特征归一 mu/sd 已算（前4000子样）；{len(ds)} 样本", flush=True)

        if args.device == "directml":
            import torch_directml
            dev = torch_directml.device()
            print(f"训练设备: DirectML ({dev})", flush=True)
        else:
            dev = torch.device("cpu")
        if args.arch == "tcn2":
            net = FrameNet2(hid=args.hid, layers=args.layers, cond=args.cond)
        else:
            if args.cond:
                raise SystemExit("GRU 骨架暂不支持条件输入，用 --arch tcn2")
            if args.device != "cpu":
                raise SystemExit("GRU 在 DirectML 上无算子支持，--arch tcn2 才能用显卡")
            net = FrameNet()
        net = net.to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        lossf = nn.MSELoss(reduction="none")
        for ep in range(args.epochs):
            t0 = time.time()
            tot = n = 0.0
            pred_low = cnt_low = 0.0     # 诊断：真值该深压帧的预测增益均值
            pred_bg = cnt_bg = 0.0       # 诊断：纯背景样本预测增益中位（应≈1）
            for xb, yb, cb in dl:
                xb = ((xb.float() - mu) / sd).clamp(-5.0, 5.0)  # v0.3 截断防爆
                yb = yb.float()
                if args.cond:
                    pred = net(xb.to(dev), cb.float().to(dev))
                else:
                    pred = net(xb.to(dev))
                yb = yb.to(dev)
                # v0.2 失衡修复：G<0.9 ×10，G<0.3 再 +10（深压帧合计 20 倍）
                lw = 1.0 + 9.0 * (yb < 0.9).float() + 10.0 * (yb < 0.3).float()
                loss = (lossf(pred, yb) * lw).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.detach())
                n += 1
                m = yb < 0.3
                if m.any():
                    pred_low += float(pred[m].mean())
                    cnt_low += 1
                mbg = (yb > 0.99).all(dim=2)   # 纯背景样本的帧 (batch, frames)
                if mbg.any():
                    pred_bg += float(pred[mbg].median())
                    cnt_bg += 1
            print(f"ep{ep:02d} loss={tot / max(n, 1):.5f} "
                  f"深压帧预测增益={pred_low / max(cnt_low, 1):.3f} "
                  f"纯背景预测增益={pred_bg / max(cnt_bg, 1):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        arch = {"type": args.arch}
        if args.arch == "tcn2":
            arch.update({"hid": args.hid, "layers": args.layers,
                         "cond": args.cond})
        torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()},
                    "mu": mu.tolist(), "sd": sd.tolist(),
                    "arch": arch}, str(ckpt))
        print(f"→ {ckpt}")

    eval_frame(ckpt, args.eval_session)


def eval_frame(ckpt, sid):
    """与 self_suppress 同口径评测：self 深压率/中位 dB vs other 深误伤率。"""
    from dsp.frame_model import FrameSuppressor
    print(f"\n== 帧级模型评测 @ {sid} ==")
    d = ROOT / "sessions" / sid
    mono = le.load_mono(d)
    fs = FrameSuppressor(ckpt)
    n_blk = 16000 * 4
    out = np.zeros_like(mono)
    for i in range(0, len(mono), n_blk):
        out[i:i + n_blk] = fs.process_block(mono[i:i + n_blk])[:len(mono[i:i + n_blk])]
    # 增益包络 = 原/处理 比
    env = np.abs(out) / (np.abs(mono) + 1e-6)
    env = np.clip(env, 0, 1.5)
    # 事件口径对比（用事件中心的包络中位）
    ev = [json.loads(l) for l in open(d / "auto_events.jsonl", encoding="utf-8")
          if json.loads(l)["type"] == "footstep"]

    def med_att(e):
        c = int(e["t"] * le.TARGET_SR)
        w = env[max(0, c - 400):c + 400]
        return -20.0 * float(np.log10(np.median(w) + 1e-9))

    def deep(e):
        c = int(e["t"] * le.TARGET_SR)
        w = env[max(0, c - 400):c + 400]
        return np.median(w) <= G_MIN * 1.5

    self_ev = [e for e in ev if e.get("self")]
    oth_ev = [e for e in ev if not e.get("self")]
    rep = {
        "session": sid, "model": str(ckpt),
        "self_deep_rate": float(np.mean([deep(e) for e in self_ev])),
        "self_median_att_db": float(np.median([med_att(e) for e in self_ev])),
        "other_deep_rate": float(np.mean([deep(e) for e in oth_ev])),
        "self_n": len(self_ev), "other_n": len(oth_ev),
    }
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    (ROOT / "sessions" / f"_frame_eval_{sid}.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    # 存 A/B 试听段
    import wave
    for tag, arr in [("frame_orig", mono), ("frame_clean", out)]:
        pcm = (np.clip(arr, -1, 1) * 32767).astype(np.int16)
        with wave.open(str(ROOT / "sessions" / sid / f"{tag}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.tobytes())
    print(f"试听: sessions/{sid}/frame_orig.wav / frame_clean.wav")


if __name__ == "__main__":
    main()

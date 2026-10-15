"""虚拟空间数据合成器：把标注过的真实片段渲染成带真值标签的训练场景。

思路（真机数据是金标准，虚拟数据负责扩量+压力测试）：
  1. extract：从边听边标的 clips 里抽"干声核"（峰能量附近 0.2-0.3s），
     按用户标的类分目录存——你的耳朵就是采样器，标过的每个片段都是一个音源。
  2. make：把核投进虚拟声场——随机方位(±80°)、距离(官方可闻域内)、
     距离增益+低通闷化(远=闷)、室内反射尾、环境噪声床、自己的声音走单声道近场。
     每个场景：session.wav + labels.jsonl(learn.py 格式) + meta.jsonl(全量真值)
     + auto_events.jsonl(引擎自己跑一遍的特征)——**虚拟会话与真实会话完全同构，
     learn.py 直接吃**。
  3. 渲染后自动对账（±0.5s），检测报告——引擎在虚拟空间里的命中率一目了然。

用法：
  python tool/virtual_scene.py extract --sessions sessions/20260927_141429 sessions/20260927_131502
  python tool/virtual_scene.py make --scenes 30
  python tool/virtual_scene.py stats
"""
import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_map import class_of  # 标注值 → 规范类统一口径（learn/extract 共用）

SR = 48000
ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = ROOT / "samples" / "virtual" / "kernels"
VIRT_DIR = ROOT / "sessions"

# 规范类 → (核目录名, 渲染角色)；标注值先过 class_of 再查这里
# 角色: enemy=空间化渲染; self=单声道近场(自己的声音)
KDIR_ROLE = {
    "footstep": ("footstep", "enemy"),
    "bot_footstep": ("bot_footstep", "enemy"),
    "gunshot": ("gunshot", "enemy"),
    "self_footstep": ("self_footstep", "self"),
    "self_gunshot": ("self_gunshot", "self"),
    "broadcast": ("broadcast", "enemy"),
    "loot": ("loot", "enemy"),
    "bot_vocal": ("bot_vocal", "enemy"),
    "melee": ("melee", "enemy"),
    "skill": ("skill", "enemy"),
    "rescue": ("rescue", "enemy"),
}
# learn.py 训练用标签名（真实类）；self 类不参与训练但保留在 meta 里
LEARN_LABEL = {"footstep": "footstep", "bot_footstep": "bot_footstep",
               "gunshot": "gunshot"}
KERNEL_LEN = {"gunshot": 0.32, "self_gunshot": 0.32,
              "footstep": 0.20, "bot_footstep": 0.20, "self_footstep": 0.20,
              "broadcast": 1.0, "loot": 0.8, "bot_vocal": 0.8, "melee": 0.4,
              "skill": 1.0, "rescue": 0.8}


# ---------- 工具 ----------

def read_wav(path):
    w = wave.open(str(path), "rb")
    sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    raw = w.readframes(n)
    w.close()
    a = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch).astype(np.float32) / 32768.0
    return a[:, :2], sr


def write_wav(path, stereo):
    stereo = np.clip(stereo, -1, 1)
    pcm = (stereo * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def one_pole_lp(x, fc):
    """一阶低通，fc 截止频率(Hz)。float64 累积，避免 int16 溢出。"""
    a = float(np.exp(-2 * np.pi * fc / SR))
    x = x.astype(np.float64)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = a * acc + (1 - a) * x[i]
        y[i] = acc
    return y


def spatial_render(kernel_mono, angle_deg, dist_m, role, rng):
    """核(单声道) → 空间化立体声事件。返回 [N,2]。

    距离模型：增益 ∝ (2.5/d)^1.15（近响远弱），远距低通闷化
    （fc = 9000*(6/d)^0.6，夹紧 1k-18k），d>8m 加两拍室内反射。
    自己的声音：单声道、最响、不做空间化（自听 foley 特性）。
    """
    x = kernel_mono.copy()
    if role == "self":
        g = rng.uniform(0.85, 1.15)
        s = np.stack([x * g, x * g], axis=1)
        return s, 0.0
    d = max(2.0, dist_m)
    g = min(1.4, (2.5 / d) ** 1.15) * rng.uniform(0.85, 1.1)
    fc = float(np.clip(9000.0 * (6.0 / d) ** 0.6, 1000, 18000))
    x = one_pole_lp(x, fc)
    # 室内反射：两拍衰减延迟，距离越远湿声占比略高
    wet = min(0.30, 0.08 + d / 200)
    tail = np.zeros(len(x) + int(0.13 * SR))
    tail[:len(x)] += x
    for delay_ms, gg in ((58, 0.5), (113, 0.28)):
        i = int(delay_ms / 1000 * SR)
        tail[i:i + len(x)] += x * gg
    x = (1 - wet) * np.pad(x, (0, len(tail) - len(x))) + wet * tail
    # ILD 声像（与 capture/synth 同 sqrt 声像律，引擎 pan 估计同口径）
    pan = np.clip(angle_deg / 80.0, -1, 1)  # 引擎角度域 ±80°
    gl = min(1.0, np.sqrt((1 - pan) / 2) * np.sqrt(2))
    gr = min(1.0, np.sqrt((1 + pan) / 2) * np.sqrt(2))
    # ITD：近耳早到 ≤0.5ms（真实感加成，引擎暂不用，先铺路）
    itd = int(abs(pan) * 0.0005 * SR)
    s = np.zeros((len(x), 2))
    if pan <= 0:  # 偏左：左耳先
        s[itd:, 0] = x[:len(x) - itd] * gl
        s[:, 1] = x * gr
    else:
        s[:, 0] = x * gl
        s[itd:, 1] = x[:len(x) - itd] * gr
    return s, pan


# ---------- extract ----------

def cut_kernel(wav_f, kdir):
    """从 clip 切干声核：峰能量 ±1/3 核长裁切、8ms 淡入淡出、静音丢弃。
    成功返回 (立体声核, )，失败返回 None。"""
    stereo, sr = read_wav(wav_f)
    mono = stereo.mean(axis=1)
    klen = int(KERNEL_LEN[kdir] * SR)
    # 峰在片段中部 ±0.5s 里找（标签窗口对准事件）
    c = len(mono) // 2
    win = mono[max(0, c - SR // 2): c + SR // 2]
    peak = np.argmax(np.convolve(win ** 2, np.ones(256) / 256, "same"))
    p = max(0, c - SR // 2) + peak
    i0 = max(0, p - klen // 3)
    seg = mono[i0:i0 + klen].copy()
    if len(seg) < klen // 2:
        return None
    seg = np.pad(seg, (0, klen - len(seg)))
    # 两端 8ms 淡入淡出防爆音
    fade = int(0.008 * SR)
    seg[:fade] *= np.linspace(0, 1, fade)
    seg[-fade:] *= np.linspace(1, 0, fade)
    if float(np.sqrt(np.mean(seg ** 2))) < 1e-4:
        return None  # 静音片段（断流/撤离后）不要
    return np.stack([seg, seg], axis=1)


def cmd_extract(args):
    out_rows = []
    for sess in args.sessions:
        ldir = Path(sess) / "labeling"
        csv_f, lab_f = ldir / "clips.csv", ldir / "user_labels.csv"
        if not (csv_f.exists() and lab_f.exists()):
            print(f"跳过 {sess}（没有标注数据）")
            continue
        labels = {}
        for line in lab_f.read_text(encoding="utf-8-sig").splitlines()[1:]:
            if line.strip():
                clip, lab = line.split(",", 1)
                labels[clip.strip()] = lab.strip()
        import csv as _csv
        with open(csv_f, encoding="utf-8-sig") as fp:
            for r in _csv.DictReader(fp):
                # 多标签：一条 clip 可贡献多个类的干声核（交叠声各自成核）
                labs = [x for x in (labels.get(r["clip"]) or "").split(";") if x]
                wav_f = ldir / "clips" / r["clip"]
                if not wav_f.exists() or not labs:
                    continue
                written = set()
                for lab in labs:
                    cls = class_of(lab)          # 新/旧/随手写标注统一成规范类
                    if cls is None or cls not in KDIR_ROLE:
                        continue
                    kdir, role = KDIR_ROLE[cls]
                    seg = cut_kernel(wav_f, kdir)
                    if seg is None:
                        continue
                    KERNEL_DIR.joinpath(kdir).mkdir(parents=True, exist_ok=True)
                    name = f"{Path(sess).name}_{r['clip']}"
                    if kdir in written:  # 同 clip 多个标签映射到同类时加后缀
                        name = f"{name}_{lab}"
                    written.add(kdir)
                    write_wav(KERNEL_DIR / kdir / name, seg)
                    out_rows.append((kdir, role, name))
    from collections import Counter
    print("抽出的干声核:", dict(Counter(k for k, _, _ in out_rows)))
    if not out_rows:
        print("没有可用核——先用边听边标标一些片段（真人/人机/自己的脚步和枪声）")


def load_kernels():
    ks = {}
    for kdir in KERNEL_DIR.iterdir() if KERNEL_DIR.exists() else []:
        fs = sorted(kdir.glob("*.wav"))
        if fs:
            ks[kdir.name] = [read_wav(f)[0].mean(axis=1) for f in fs]
    return ks


# ---------- make ----------

def cmd_make(args):
    rng = np.random.default_rng(args.seed)
    kernels = load_kernels()
    need = ["footstep", "bot_footstep", "gunshot"]
    missing = [k for k in need if k not in kernels]
    if missing:
        raise SystemExit(f"缺核: {missing}——先 extract")
    self_fs = kernels.get("self_footstep", [])
    self_gs = kernels.get("self_gunshot", [])

    from dsp.pipeline import Engine
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    engine = Engine(cfg)

    outdir = VIRT_DIR / f"virtual_{args.seed:04d}"
    outdir.mkdir(parents=True, exist_ok=True)
    labels_f = open(outdir / "labels.jsonl", "w", encoding="utf-8")
    meta_f = open(outdir / "meta.jsonl", "w", encoding="utf-8")
    auto_f = open(outdir / "auto_events.jsonl", "w", encoding="utf-8")

    total_ev, hit, type_hit = 0, 0, 0
    t_off = 0.0  # 会话全局时间轴：各场景时间依次顺延，labels/meta/auto 才能互相对上
    for sc in range(args.scenes):
        dur = rng.uniform(args.min_dur, args.max_dur)
        n = int(dur * SR)
        bed = np.convolve(rng.standard_normal(n), np.ones(400) / 400, "same") \
            * rng.uniform(0.002, 0.005)
        scene_l, scene_r = bed.copy(), bed.copy()
        # 事件调度：间隔≥0.3s，枪声后 1.1s 内不放脚步（引擎掩蔽窗会吞，放了就是对不齐的标签）
        events, t = [], rng.uniform(0.5, 1.5)
        last_gun = -9.0
        while t < dur - 1.0:
            r = rng.random()
            if r < 0.06 and self_gs:
                events.append((t, "self_gunshot", "self", 0.0))
                last_gun = t
            elif r < 0.32 and self_fs:
                events.append((t, "self_footstep", "self", 0.0))
            elif r < 0.50:
                events.append((t, "gunshot", "enemy",
                               float(rng.uniform(10, 80))))
                last_gun = t
            elif r < 0.75:
                events.append((t, "bot_footstep", "enemy",
                               float(rng.uniform(2, 45))))
            else:
                events.append((t, "footstep", "enemy",
                               float(rng.uniform(2, 45))))
            gap = 0.10 if events[-1][1] in ("gunshot", "self_gunshot") \
                and rng.random() < 0.5 else rng.uniform(0.3, 1.6)
            t += gap
        # 静步变体：一部分脚步降增益（引擎 quiet 档的来源）
        for i, (et, name, role, d) in enumerate(events):
            if name in ("footstep", "bot_footstep") and rng.random() < 0.2:
                events[i] = (et, name, role, d, 0.18)  # 增益乘数
            else:
                events[i] = (et, name, role, d, 1.0)

        truth_rows = []
        for et, name, role, d, mul in events:
            kern = kernels[name][rng.integers(0, len(kernels[name]))]
            ang = float(rng.uniform(-80, 80)) if role == "enemy" else 0.0
            s, pan = spatial_render(kern * mul, ang, d, role, rng)
            i = int(et * SR)
            if i + len(s) > n:
                s = s[:n - i]
            scene_l[i:i + len(s)] += s[:, 0]
            scene_r[i:i + len(s)] += s[:, 1]
            gt = t_off + et  # 全局时间
            learn = LEARN_LABEL.get(name)
            if learn:
                labels_f.write(json.dumps({"t": round(gt, 3), "label": learn},
                                          ensure_ascii=False) + "\n")
            meta_f.write(json.dumps({"t": round(gt, 3), "type": name, "role": role,
                                     "angle": round(ang, 1), "dist": round(d, 1),
                                     "pan": round(pan, 3), "quiet": mul < 1.0},
                                    ensure_ascii=False) + "\n")
            truth_rows.append({"t": gt, "type": name})

        wav_path = outdir / f"scene_{sc:03d}.wav"
        scene = np.stack([scene_l, scene_r], axis=1)
        scene /= max(1e-6, np.abs(scene).max()) / 0.95
        write_wav(wav_path, scene)

        # 引擎自己跑一遍（离线快速，不吃实时等待）
        from capture.source import FileSource
        eng = Engine(cfg)
        src = FileSource(str(wav_path), sr=cfg["sample_rate"],
                         block=cfg["block_size"], speed=1e9)
        det = []
        while True:
            item = src.read()
            if item is None:
                break
            t, frame = item
            snap = eng.process(t, frame)
            ev = snap.get("event")
            if ev and ev["type"] in ("footstep", "gunshot", "bot_footstep",
                                      "glass", "action"):
                row = {"t": round(t_off + ev["t"], 3), "type": ev["type"],
                       "pan": round(ev["pan"], 3), "self": bool(ev.get("self")),
                       "self_reason": ev.get("self_reason", ""),
                       "feat": {k: round(float(snap["feats"][k]), 6) for k in
                                ("low", "mid", "high", "low_ratio", "high_ratio",
                                 "corr", "low_flat", "rms")}}
                det.append(row)
                auto_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # 对账：learn.py 只要求时间上 ±0.5s 内有检出（用户标签是真值，
        # 检出类型不一致不影响学习）；类型一致率单列作参考
        truth = [m for m in truth_rows if m["type"] in LEARN_LABEL]
        for m in truth:
            total_ev += 1
            if any(abs(d0["t"] - m["t"]) <= 0.5 for d0 in det):
                hit += 1
                if any(abs(d0["t"] - m["t"]) <= 0.5 and
                       d0["type"] == LEARN_LABEL[m["type"]] for d0 in det):
                    type_hit += 1
        print(f"场景 {sc + 1}/{args.scenes}: {len(events)} 事件, "
              f"引擎检出 {len(det)} 条")
        t_off += dur

    labels_f.close(); meta_f.close(); auto_f.close()
    print(f"\n虚拟会话: {outdir}")
    print(f"时间对账: {hit}/{total_ev} ({hit / max(1, total_ev) * 100:.0f}%) "
          f"真值 ±0.5s 内有检出（learn.py 要求）")
    print(f"类型对账: {type_hit}/{total_ev} "
          f"({type_hit / max(1, total_ev) * 100:.0f}%) 检出类型与真值一致（参考）")
    print("可直接喂: python tool/learn.py --stats")


def cmd_stats(args):
    ks = load_kernels()
    for k, v in sorted(ks.items()):
        lens = [len(x) for x in v]
        print(f"{k:<16} {len(v)} 核, 平均 {np.mean(lens) / SR * 1000:.0f}ms")


def main():
    ap = argparse.ArgumentParser(description="虚拟空间数据合成器")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="从标注 clips 抽干声核")
    p.add_argument("--sessions", nargs="+", required=True)
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("make", help="渲染虚拟场景会话")
    p.add_argument("--scenes", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-dur", type=float, default=20.0)
    p.add_argument("--max-dur", type=float, default=40.0)
    p.set_defaults(fn=cmd_make)

    p = sub.add_parser("stats", help="看核库")
    p.set_defaults(fn=cmd_stats)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

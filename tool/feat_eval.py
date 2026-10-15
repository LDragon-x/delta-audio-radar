# -*- coding: utf-8 -*-
"""特征烘焙赛：同一批复标样本，多特征集对比跨局留出 5-NN。

任务A 人机(bot_footstep) vs 真人(footstep)——三次撞墙的主战场。
任务B 自己(self_footstep) vs 敌人(footstep/bot_footstep)——消自己的
      ML 可行性（防线链之外的第二 opinion）。

特征集：
  base8       v0.1 引擎特征（对照组）
  mel_mean    64 带 log-mel 时间均值
  mel_meanstd 均值+std 128 维
  mel_mod32   时序调制谱（脚步纹理）
  all         mel_meanstd + mel_mod32 拼接

诚实口径：按局留出（训练见不到测试局的任何样本），5-NN 多数表决。
打印每任务每特征集的检出率/误伤率，不作模型上线的依据，只回答
"哪个特征方向值得在 v0.2 里深挖"。

用法：
  python tool/feat_eval.py
"""
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tool"))
from label_map import class_of  # noqa: E402
from dsp.logmel import logmel_patch, pcen_patch, pooled  # noqa: E402

FEAT_KEYS = ["low", "mid", "high", "low_ratio", "high_ratio", "corr",
             "low_flat", "rms"]
LOG_KEYS = {"low", "mid", "high", "rms"}
TARGET_SR = 16000


def base8(f):
    return np.array([float(np.log(f[k] + 1e-9)) if k in LOG_KEYS else float(f[k])
                     for k in FEAT_KEYS])


def load_mono(session_dir):
    """session.wav → 16k 单声道（懒加载缓存 labeling/_mono16k.npy）。"""
    cache = Path(session_dir) / "labeling" / "_mono16k.npy"
    if cache.exists():
        return np.load(cache)
    import wave
    with wave.open(str(Path(session_dir) / "session.wav"), "rb") as w:
        sr, ch, sw, n = (w.getframerate(), w.getnchannels(),
                         w.getsampwidth(), w.getnframes())
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch).mean(axis=1)
    # 48k→16k 直接抽稀（验证够用；引擎上线再换抗混叠）
    a = a[::(sr // TARGET_SR)].astype(np.float32) / 32768.0
    np.save(cache, a)
    return a


def collect():
    """→ [(tag, session, vec_dict, taskA_label, taskB_label)]"""
    out = []
    for d in sorted(glob.glob(str(ROOT / "sessions" / "2*"))):
        af, cf, uf = (Path(d) / "auto_events.jsonl",
                      Path(d) / "labeling" / "clips.csv",
                      Path(d) / "labeling" / "user_labels.csv")
        if not (af.exists() and cf.exists() and uf.exists()):
            continue
        events = [json.loads(l) for l in af.read_text(encoding="utf-8").splitlines()
                  if l.strip()]
        tmap = {r["clip"]: float(r["t"])
                for r in csv.DictReader(open(cf, encoding="utf-8-sig"))}
        mono = None
        for r in csv.DictReader(open(uf, encoding="utf-8-sig")):
            if not r.get("label"):
                continue
            cls_set = {class_of(l.strip()) for l in r["label"].split(";")}
            if not ({"footstep", "bot_footstep", "self_footstep"} & cls_set):
                continue
            t = tmap.get(r["clip"])
            if t is None:
                continue
            near = [e for e in events
                    if abs(e["t"] - t) <= 0.5 and "feat" in e]
            if not near:
                continue
            e = min(near, key=lambda x: abs(x["t"] - t))
            task_a = ("bot" if "bot_footstep" in cls_set
                      else "hum" if "footstep" in cls_set else None)
            task_b = ("self" if "self_footstep" in cls_set
                      else "foe" if task_a else None)
            if mono is None:
                mono = load_mono(d)
            patch = logmel_patch(mono, TARGET_SR, e["t"])
            pv = pooled(patch)
            pv_pcen = pooled(pcen_patch(mono, TARGET_SR, e["t"]))
            vecs = {"base8": base8(e["feat"]),
                    "mel_mean": pv["mel_mean"],
                    "mel_meanstd": pv["mel_meanstd"],
                    "mel_mod32": pv["mel_mod32"],
                    "all": np.concatenate([pv["mel_meanstd"],
                                           pv["mel_mod32"]]),
                    "pcen_mean": pv_pcen["mel_mean"],
                    "pcen_meanstd": pv_pcen["mel_meanstd"],
                    "pcen_mod32": pv_pcen["mel_mod32"],
                    "pcen_all": np.concatenate([pv_pcen["mel_meanstd"],
                                                pv_pcen["mel_mod32"]])}
            out.append((Path(d).name, vecs, task_a, task_b))
    return out


def loo_eval(samples, feat_key, label_of):
    """按局留出 5-NN。返回 (检出率, 误伤率, n_pos, n_neg)。"""
    helds = sorted(set(s[0] for s in samples))
    tp = fp = fn = tn = 0
    for held in helds:
        train = [(s[1][feat_key], label_of(s)) for s in samples
                 if s[0] != held and label_of(s)]
        test = [s for s in samples if s[0] == held and label_of(s)]
        for s in test:
            v = s[1][feat_key]
            lab = label_of(s)
            near = sorted(train, key=lambda ts: float(np.linalg.norm(ts[0] - v)))[:5]
            pred_pos = sum(1 for _, l in near if l == "pos") >= 3
            if pred_pos and lab == "pos":
                tp += 1
            elif pred_pos:
                fp += 1
            elif lab == "pos":
                fn += 1
            else:
                tn += 1
    n_pos, n_neg = tp + fn, fp + tn
    return (tp / max(n_pos, 1), fp / max(n_neg, 1), n_pos, n_neg)


def main():
    samples = collect()
    print(f"样本 {len(samples)} 条（按局留出）")

    tasks = [("A 人机vs真人",
              lambda s: ("pos" if s[2] == "bot" else
                         "neg" if s[2] == "hum" else None)),
             ("B 自己vs敌人",
              lambda s: ("pos" if s[3] == "self" else
                         "neg" if s[3] == "foe" else None))]
    for title, label_of in tasks:
        print(f"\n=== 任务{title} ===")
        print(f"{'特征集':<14}{'检出率':>8}{'误伤率':>8}  样本")
        for fk in ["base8", "mel_mean", "mel_meanstd", "mel_mod32", "all",
                   "pcen_mean", "pcen_meanstd", "pcen_mod32", "pcen_all"]:
            det, false, n_pos, n_neg = loo_eval(samples, fk, label_of)
            flag = ""
            if det > 0.75 and false < 0.15:
                flag = "  ← 可用！"
            elif det > 0.65 and false < 0.25:
                flag = "  ← 有戏"
            print(f"{fk:<14}{det:>8.0%}{false:>8.0%}  {n_pos}pos/{n_neg}neg{flag}")
    print("\n口径说明：检出率=该类被正确认出比例，误伤率=另一类被错杀比例。")


if __name__ == "__main__":
    main()

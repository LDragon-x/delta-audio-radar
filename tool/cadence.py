# -*- coding: utf-8 -*-
"""节奏分析 v1：人机是节拍器，真人是变速的——步序间隔规律性对比。

思路：把一局里的脚步事件（剔除 self）按时间串成"步序run"（相邻间隔
≤GAP 归同一段走路），对每段算规律性指标；用边听边标的人工标签
（footstep=真人 / bot_footstep=人机）给 run 定真值，看两类在节奏上
分不分得开。分得开 → 引擎分类器多一个免费特征；分不开也是结论。

指标（每 run）：
  n        步数
  med_iv   间隔中位数（步频=1/med_iv）
  cv       间隔标准差/均值（变异系数）
  jitter   median(|I-中位|)/中位（节拍抖动，越小越机械）
  reg20    间隔落在±20%中位内的比例（节拍命中率）
  pan_drift 相邻步声像跳变中位数（真人横移多，人机直线走）

用法：
  python tool/cadence.py [--gap 2.5] [--min 4]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_map import class_of

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "sessions"


def load_sessions():
    """→ {session_dir: (events, labeled_t)}；labeled_t: {class: [t,...]}。"""
    out = {}
    for d in sorted(SESSIONS.glob("*")):
        af = d / "auto_events.jsonl"
        if not af.exists():
            continue
        events = [json.loads(x) for x in af.read_text(encoding="utf-8").splitlines()
                  if x.strip()]
        labeled = {"footstep": [], "bot_footstep": []}
        cf, uf = d / "labeling" / "clips.csv", d / "labeling" / "user_labels.csv"
        if cf.exists() and uf.exists():
            tmap = {}
            with open(cf, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    tmap[r["clip"]] = float(r["t"])
            with open(uf, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    t = tmap.get(r["clip"])
                    if t is None or not r.get("label"):
                        continue
                    cls_seen = set()
                    for lab in r["label"].split(";"):
                        cls = class_of(lab.strip())
                        if cls in ("footstep", "bot_footstep") and cls not in cls_seen:
                            cls_seen.add(cls)
                            labeled[cls].append(t)
        out[d] = (events, labeled)
    return out


def runs_from(times_pans, gap):
    """times_pans: 按 t 排序的 [(t, pan)] → 步序 run 列表（每 run 指标 dict）。"""
    runs, cur = [], []
    for tp in times_pans:
        if cur and tp[0] - cur[-1][0] > gap:
            runs.append(cur)
            cur = []
        cur.append(tp)
    if cur:
        runs.append(cur)
    feats = []
    for run in runs:
        if len(run) < 4:
            continue
        ts = np.array([t for t, _ in run])
        pans = np.array([p for _, p in run])
        iv = np.diff(ts)
        med = float(np.median(iv))
        if med <= 0:
            continue
        feats.append({
            "n": len(run),
            "med_iv": med,
            "freq": 1.0 / med,
            "cv": float(iv.std() / iv.mean()),
            "jitter": float(np.median(np.abs(iv - med)) / med),
            "reg20": float(np.mean(np.abs(iv - med) <= 0.2 * med)),
            "pan_drift": float(np.median(np.abs(np.diff(pans)))),
            "t0": float(ts[0]), "t1": float(ts[-1]),
        })
    return feats


def auc(scores, labels):
    """ Mann-Whitney AUC：scores 越大越像 labels==1。 """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=2.5, help="步序切分间隔(秒)")
    ap.add_argument("--min", type=int, default=4, help="run 最少步数")
    args = ap.parse_args()

    data = load_sessions()
    rows = []   # (feat, truth)  truth: 1=真人 0=人机
    for d, (events, labeled) in data.items():
        steps = sorted((e["t"], float(e.get("pan", 0)))
                       for e in events
                       if e["type"] in ("footstep", "bot_footstep")
                       and not e.get("self"))
        for f in runs_from(steps, args.gap):
            if f["n"] < args.min:
                continue
            has_h = any(f["t0"] - 0.5 <= t <= f["t1"] + 0.5
                        for t in labeled["footstep"])
            has_b = any(f["t0"] - 0.5 <= t <= f["t1"] + 0.5
                        for t in labeled["bot_footstep"])
            if has_h == has_b:      # 无标签或两类混在同一段，弃
                continue
            rows.append((f, 1 if has_h else 0))

    if len(rows) < 10:
        print(f"带真值的步序 run 只有 {len(rows)} 段（<10），先多标几局再验")
        return

    print(f"带真值步序 run: {len(rows)} 段"
          f"（真人 {sum(t for _, t in rows)} / 人机 {sum(1 - t for _, t in rows)}）")
    print(f"{'指标':<10}{'真人中位':>10}{'人机中位':>10}{'AUC':>7}  方向")
    for key, human_high in [("n", None), ("med_iv", None), ("freq", None),
                            ("cv", True), ("jitter", True), ("reg20", False),
                            ("pan_drift", True)]:
        s = [f[key] for f, _ in rows]
        lab = [t for _, t in rows]
        h = [f[key] for f, t in rows if t == 1]
        b = [f[key] for f, t in rows if t == 0]
        a = auc(s, lab)
        # 方向翻转成"值大=真人"
        a_disp = a if human_high else 1 - a
        direction = "大=真人" if a >= 0.5 else "小=真人"
        print(f"{key:<10}{np.median(h):>10.3f}{np.median(b):>10.3f}"
              f"{a_disp:>7.3f}  {direction}")

    # 组合：jitter+pan_drift 二维阈值粗扫（线性评分 z 分数相加）
    def z(v):
        v = np.array(v, dtype=float)
        return (v - v.mean()) / (v.std() + 1e-9)
    combo = z([f["jitter"] for f, _ in rows]) + z([f["pan_drift"] for f, _ in rows])
    print(f"\n组合(jitter+pan_drift) AUC: {auc(list(combo), [t for _, t in rows]):.3f}")
    print("结论：AUC>0.75 值得加进分类器特征；0.6~0.75 辅助特征；<0.6 节奏不可分")


if __name__ == "__main__":
    main()

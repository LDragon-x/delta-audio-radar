# -*- coding: utf-8 -*-
"""加菜包：往已有标注包里追加"用户点名的值钱段"。

两类 bonus clips：
  act_*  战后偷听窗——交战回合结束后 3~20 秒内最响的 action 事件（打药/
         搜物资/杂声，劝架时机的金标准），每回合最多 2 个，全局封顶 --cap
  run_*  完整人机步——连续步序(≥--min 步,间隔≤2.5s)最长的前 --runs 段，
         切整段中间 8 秒（节奏分析要的就是成串的），标 人机脚步 即真值

往 clips.csv 尾部追加行（已标行不受影响），wav 进 clips/。
labeling_ui 启动时一次性读 clips.csv，追加后要重启服务才显示。

用法：
  python tool/bonus_clips.py --session sessions/<时间戳> [--cap 20] [--runs 3]
"""
import argparse
import csv
import json
import wave
from pathlib import Path

CLIP_PRE = 1.2
CLIP_POST = 1.2
RUN_LEN = 8.0       # run_* 片段时长（秒）


def load_wav(path):
    w = wave.open(str(path), "rb")
    sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    data = w.readframes(n)
    w.close()
    return data, sr, ch, sw, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--cap", type=int, default=20, help="act_ 片段全局上限")
    ap.add_argument("--runs", type=int, default=3, help="run_ 长段条数")
    ap.add_argument("--min", type=int, default=10, help="run 最少步数")
    args = ap.parse_args()

    sdir = Path(args.session)
    out = sdir / "labeling"
    clips_dir = out / "clips"
    data, sr, ch, sw, n = load_wav(sdir / "session.wav")
    events = [json.loads(l) for l in
              (sdir / "auto_events.jsonl").read_text(encoding="utf-8").splitlines()
              if l.strip()]

    import numpy as np
    raw = np.frombuffer(data[: n * ch * sw], dtype=np.int16).reshape(-1, ch)

    def write_clip(t, pre, post, name):
        i0 = max(0, int((t - pre) * sr))
        i1 = min(len(raw), int((t + post) * sr))
        with wave.open(str(clips_dir / name), "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(sw)
            w.setframerate(sr)
            w.writeframes(raw[i0:i1].tobytes())

    guns = sorted(e["t"] for e in events if e["type"] == "gunshot")
    eps = []
    for t in guns:
        if eps and t - eps[-1][-1] < 4.0:
            eps[-1].append(t)
        else:
            eps.append([t])

    # act_*：战后窗内最响 action 事件
    acts = []
    for ep in eps:
        cands = [e for e in events
                 if e["type"] == "action" and ep[-1] + 3 <= e["t"] <= ep[-1] + 20]
        cands.sort(key=lambda e: -float(e.get("feat", {}).get("rms", 0)))
        acts.extend(cands[:2])
    acts.sort(key=lambda e: -float(e.get("feat", {}).get("rms", 0)))
    acts = acts[: args.cap]

    # run_*：最长连续步序段
    steps = sorted(e["t"] for e in events
                   if e["type"] in ("footstep", "bot_footstep")
                   and not e.get("self"))
    runs, cur = [], []
    for t in steps:
        if cur and t - cur[-1] > 2.5:
            runs.append(cur)
            cur = []
        cur.append(t)
    if cur:
        runs.append(cur)
    runs = [r for r in runs if len(r) >= args.min]
    runs.sort(key=len, reverse=True)
    chosen_runs = []
    for r in runs:
        if len(chosen_runs) >= args.runs:
            break
        if any(abs(r[0] - c[0]) < 20 for c in chosen_runs):
            continue  # 段间去重
        chosen_runs.append(r)

    rows, names = [], set()
    for e in acts:
        name = f"act_{int(e['t']):04d}s.wav"
        if name in names:
            continue
        names.add(name)
        write_clip(e["t"], CLIP_PRE, CLIP_POST, name)
        f = e.get("feat", {})
        rows.append({"clip": name, "t": round(e["t"], 2), "type": "action",
                     "auto_self": bool(e.get("self")), "self_reason": "",
                     "corr": f.get("corr", ""), "rms": f.get("rms", ""),
                     "priority": "加菜:战后打药窗",
                     "label": "", "note": "战后偷听窗:打药/搜物资/杂声"})
    for r in chosen_runs:
        mid = (r[0] + r[-1]) / 2
        name = f"run_{int(r[0]):04d}s.wav"
        if name in names:
            continue
        names.add(name)
        write_clip(mid, RUN_LEN / 2, RUN_LEN / 2, name)
        rows.append({"clip": name, "t": round(mid, 2), "type": "footstep",
                     "auto_self": False, "self_reason": "",
                     "corr": "", "rms": "",
                     "priority": f"加菜:连续{len(r)}步",
                     "label": "", "note": f"成串脚步段({len(r)}步,"
                                           f"{r[-1]-r[0]:.0f}s) 建议标人机/真人脚步"})
        print(f"run: {r[0]:.0f}-{r[-1]:.0f}s {len(r)}步")

    csv_path = out / "clips.csv"
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writerows(rows)
    print(f"加菜 {len(rows)} 段（act {len(acts)} / run {len(chosen_runs)}），"
          f"clips.csv 现有 {sum(1 for _ in open(csv_path, encoding='utf-8-sig')) - 1} 行")


if __name__ == "__main__":
    main()

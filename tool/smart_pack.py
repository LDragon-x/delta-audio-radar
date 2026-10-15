"""智能优先级资料包：把"最值得人工精标"的段落排最前。

v2 选段逻辑（用户实战反馈改）：交火最密的时候全是枪声，听不出脚步；
真正值钱的是**交战开始前 5~10 秒**——摸近/蹲点/换位的听声辨位阶段。
  1. 交战回合：相邻枪声(间隔<4s)归一回合，回合起点=第一声枪响
  2. 候选=脚步事件（剔除已判 self 的）：落在任一回合起点前 5~10s 的
     得"黄金窗"加成；±2.5s 内有枪声或自己开火（鼠标左键边沿）的重罚沉底
  3. 步序加成：±2s 内连续脚步越多越值钱（节奏分析要的正是成串的脚步）
  4. 非极大值抑制（±1.2s）防同一段占满名额；--random 条均匀保底。

剪辑命名与 make_labeling_pack 完全同口径（type_<秒>_self|unk.wav），
已标过的 clip 自动保留并排最前，标注不丢。

用法：
  python tool/smart_pack.py --session sessions/<时间戳> [--top 80] [--random 15]
"""
import argparse
import csv
import json
import random
import sys
import wave
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLIP_PRE = 1.2
CLIP_POST = 1.2
WIN = 1.5      # 复杂度统计窗（秒）
NMS = 1.2      # 非极大值抑制窗（秒）


def load_wav(path):
    w = wave.open(str(path), "rb")
    sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    data = w.readframes(n)
    w.close()
    return data, sr, ch, sw, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--top", type=int, default=80, help="优先级片段数")
    ap.add_argument("--random", type=int, default=15, help="随机保底样本数")
    ap.add_argument("--push", type=int, default=40,
                    help="突进搜人段片段上限（全局按响度取）")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import numpy as np
    sdir = Path(args.session)
    out = sdir / "labeling"
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    data, sr, ch, sw, n = load_wav(sdir / "session.wav")

    events = [json.loads(l) for l in
              (sdir / "auto_events.jsonl").read_text(encoding="utf-8").splitlines()
              if l.strip()]
    raw = np.frombuffer(data[: n * ch * sw], dtype=np.int16).reshape(-1, ch)
    amp = np.abs(raw).max(axis=1).astype(np.float32) / 32768.0
    sec_amp = amp[: (len(amp) // sr) * sr].reshape(-1, sr).mean(axis=1)
    silent_sec = set(int(i) for i in np.where(sec_amp < 1e-4)[0])

    keys = []
    kf = sdir / "keys.jsonl"
    if kf.exists():
        keys = [json.loads(l) for l in kf.read_text(encoding="utf-8").splitlines()
                if l.strip()]

    def key_count(t):
        return sum(1 for k in keys if abs(k["t"] - t) <= WIN)

    evs = [e for e in events if int(e["t"]) not in silent_sec
           and e["type"] in ("footstep", "bot_footstep", "gunshot", "glass",
                             "action")]
    # 选段只从脚步里挑（听声辨位的料），枪声只用于定位交战回合
    steps = [e for e in evs if e["type"] in ("footstep", "bot_footstep")
             and not e.get("self")]
    guns = sorted(e["t"] for e in evs if e["type"] == "gunshot")
    fire_t = [k["t"] for k in keys if k.get("down") and k["key"] == "left"]

    # 交战回合：相邻枪声(间隔<4s)归一回合，回合起点=第一声枪响
    episodes = []
    for t in guns:
        if episodes and t - episodes[-1][-1] < 4.0:
            episodes[-1].append(t)
        else:
            episodes.append([t])
    PRE_LO, PRE_HI = 10.0, 5.0   # 黄金窗：交战开始前 5~10 秒

    def clean_window(t, margin=2.5):
        return not any(abs(g - t) <= margin for g in guns) \
            and not any(abs(f - t) <= margin for f in fire_t)

    scored = []
    for e in steps:
        t = e["t"]
        seq = sum(1 for o in steps if o is not e and abs(o["t"] - t) <= 2.0)
        golden = any(ep[0] - PRE_HI >= t >= ep[0] - PRE_LO for ep in episodes)
        score = (3.0 * seq                                  # 成串的脚步（节奏料）
                 + 40.0 * float(e.get("feat", {}).get("rms", 0))
                 + 0.15 * key_count(t)
                 + 30.0 * (1 if golden else 0)               # 交战前黄金窗：压倒性优先
                 - (10.0 if not clean_window(t) else 0.0))  # 枪声/自己开火旁沉底
        scored.append((score, e))
    scored.sort(key=lambda x: -x[0])

    picked = []
    used = []
    for score, e in scored:
        if len(picked) >= args.top:
            break
        if any(abs(e["t"] - t0) < NMS for t0 in used):
            continue
        picked.append((score, e, None))
        used.append(e["t"])

    # 突进搜人段（重要负样本）：打完一个敌人→开技能→键盘密集移动找剩下的
    # 人，这段时间自己脚步特别响又压不掉（WASD 掩蔽窗仅 0.3s）。识别：
    # 交战回合结束后 25s 内键盘移动边沿密集(≥6)，到下一回合枪响为止；
    # 段内选 self 标记或单声道相关(居中=自己)的响脚步，每段最多 4 个。
    move_t = [k["t"] for k in keys if k.get("down") and k["key"] in
              ("w", "a", "s", "d", "space", "c")]
    all_steps = [e for e in evs if e["type"] in ("footstep", "bot_footstep")]

    def pick_push(lo, hi, cap=4):
        """段内选 self 标记或单声道相关(居中=自己)的响脚步，每段最多 cap 个。"""
        cands = sorted((e for e in all_steps if lo <= e["t"] <= hi),
                       key=lambda e: -float(e.get("feat", {}).get("rms", 0)))
        taken = 0
        for e in cands:
            if taken >= cap:
                break
            if not (e.get("self") or e.get("feat", {}).get("corr", 0) >= 0.9):
                continue  # 只要自己的脚步
            if any(abs(e["t"] - t0) < NMS for t0 in used + push_used):
                continue
            push_named.append((0.0, e, f"push_{int(e['t']):04d}s.wav"))
            push_used.append(e["t"])
            taken += 1

    push_named = []
    push_used = []
    for ep in episodes:
        end = ep[-1]
        nxt = min((g[0] for g in episodes if g[0] > end + 1.0), default=end + 30.0)
        lo, hi = end + 1.0, min(end + 30.0, nxt - 1.0)
        if sum(1 for t in move_t if lo <= t <= hi) < 6:
            continue
        pick_push(lo, hi)

    # 技能段（用户亲授：按 X 开技能后的大脚步全是自己）——按 X 边沿起 12 秒
    # 窗口直接收，不依赖交战回合，比之前"键盘边沿≥6"的猜测可靠得多
    for st in [k["t"] for k in keys if k.get("down") and k["key"] == "x"]:
        pick_push(st + 0.3, st + 12.0)

    # 突进段全局封顶：按响度取最值钱的 --push 个，防淹没黄金窗片段
    if len(push_named) > args.push:
        keep, kused = [], []
        for _, e, n in sorted(push_named,
                              key=lambda x: -float(x[1].get("feat", {}).get("rms", 0))):
            if len(keep) >= args.push:
                break
            if any(abs(e["t"] - t0) < NMS for t0 in kused):
                continue
            keep.append((0.0, e, n))
            kused.append(e["t"])
        push_named = keep
        push_used = kused

    # 随机保底：未被优先级选中的脚步，均匀抽保持分布覆盖
    rng = random.Random(args.seed)
    chosen_t = {e["t"] for _, e, _ in picked} | set(push_used)
    rest = [e for e in steps if e["t"] not in chosen_t]
    rng.shuffle(rest)
    base = picked[:] + push_named
    used2 = used + push_used
    for e in rest:
        if len(base) >= args.top + args.random + len(push_named):
            break
        if any(abs(e["t"] - t0) < NMS for t0 in used2):
            continue
        base.append((0.0, e, None))
        used2.append(e["t"])

    # 已标过的 clip 保留并置顶（标注进度不丢）
    uf = out / "user_labels.csv"
    labeled = set()
    if uf.exists():
        with open(uf, encoding="utf-8-sig") as f:
            labeled = {r["clip"] for r in csv.DictReader(f) if r.get("label")}

    def clip_name(e):
        return f"{e['type']}_{int(e['t']):04d}s_{'self' if e.get('self') else 'unk'}.wav"

    base_named = [(s, e, n or clip_name(e)) for s, e, n in base]
    labeled_es = [(s, e, n) for s, e, n in base_named if n in labeled]
    unlabeled = [(s, e, n) for s, e, n in base_named if n not in labeled]
    final = labeled_es + unlabeled

    rows = []
    for rank, (score, e, name) in enumerate(final):
        t = e["t"]
        i0 = max(0, int((t - CLIP_PRE) * sr))
        i1 = min(len(raw), int((t + CLIP_POST) * sr))
        with wave.open(str(clips_dir / name), "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(sw)
            w.setframerate(sr)
            w.writeframes(raw[i0:i1].tobytes())
        f = e.get("feat", {})
        rows.append({"clip": name, "t": round(t, 2), "type": e["type"],
                     "auto_self": bool(e.get("self")),
                     "self_reason": e.get("self_reason", ""),
                     "corr": f.get("corr", ""), "rms": f.get("rms", ""),
                     "priority": "已标" if name in labeled else round(score, 1),
                     "label": "", "note": ""})

    with open(out / "clips.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    (out / "README.txt").write_text(
        "边听边标·智能优先级包\n"
        "====================\n"
        f"{len(rows)} 个片段：靠前的是【交战开始前 5~10 秒】的成串脚步"
        f"（听声辨位黄金窗）；push_ 开头的是【突进/技能段】——打完敌人后\n"
        "开技能猛按键盘找人的那段时间、以及按 X 开技能后的 12 秒，自己的\n"
        "响脚步（引擎最难压的负样本），这类标 自己→突进脚步；枪声/自己开火\n"
        "附近的声音片段已沉底剔除。\n"
        "priority 列是打分（已标过的置顶显示'已标'）。\n"
        "打开 labeling_ui 网页标注：点大类/小类=挂标签，再点=取消，"
        "同段可挂多类。\n",
        encoding="utf-8")
    n_lab = len(labeled_es)
    print(f"优先级包: {out}  {len(rows)} clips"
          f"（已标保留 {n_lab}，优先级 {len(unlabeled)}，突进段 {len(push_named)}）")
    top_types = defaultdict(int)
    for _, e, _ in picked[:10]:
        top_types[e["type"]] += 1
    print("TOP10 片段类型:", dict(top_types))


if __name__ == "__main__":
    main()

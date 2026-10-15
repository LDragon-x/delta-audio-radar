"""把采集会话整理成"可人工标注"的资料包：短片段 + 清单 CSV。

用户戴着耳机听 clip，在 CSV 里填 label 列（值随意，如 enemy_footstep /
bot_footstep / self_footstep / self_gunshot / ...），我再用 ingest 工具吃回去。

用法：
  python tool/make_labeling_pack.py --session sessions/<时间戳> [--per-type 40]
输出：
  sessions/<ts>/labeling/clips/<类型>_<秒数>_<self标记>.wav   （事件前后 ±1.2s）
  sessions/<ts>/labeling/clips.csv                             （清单，label/note 两列留空给用户填）
  sessions/<ts>/labeling/README.txt                            （怎么标、怎么给我）
"""
import argparse
import csv
import json
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLIP_PRE = 1.2   # 片段：事件前 1.2s ~ 后 1.2s（2.4s 足够听清来源）
CLIP_POST = 1.2


def load_wav(path):
    w = wave.open(str(path), "rb")
    sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    data = w.readframes(n)
    w.close()
    return data, sr, ch, sw, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--per-type", type=int, default=40, help="每类事件抽样数")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sdir = Path(args.session)
    out = sdir / "labeling"
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    data, sr, ch, sw, n = load_wav(sdir / "session.wav")

    events = [json.loads(l) for l in (sdir / "auto_events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # 排除打无声期的（撤离后菜单段）
    import numpy as np
    raw = np.frombuffer(data[: n * ch * sw], dtype=np.int16).reshape(-1, ch)
    amp = np.abs(raw).max(axis=1).astype(np.float32) / 32768.0
    sec_amp = amp[: (len(amp) // sr) * sr].reshape(-1, sr).mean(axis=1)  # 每秒平均幅度
    silent_sec = set(int(i) for i in np.where(sec_amp < 1e-4)[0])

    # 分层抽样：每类按时间均匀取，自带 self 标记的全保留名额内
    by_type = defaultdict(list)
    for e in events:
        by_type[e["type"]].append(e)
    import random
    rng = random.Random(args.seed)
    picked = []
    for ty, es in sorted(by_type.items()):
        es = [e for e in es if int(e["t"]) not in silent_sec]
        rng.shuffle(es)
        # self 和非 self 各占一半配额，两边都听才有对照
        self_es = [e for e in es if e.get("self")]
        oth_es = [e for e in es if not e.get("self")]
        half = args.per_type // 2
        grp = sorted(self_es[:half] + oth_es[:half + args.per_type % 2], key=lambda e: e["t"])
        picked.extend(grp)
        print(f"{ty:<10} 共 {len(es):5d} 条（self {len(self_es)}）→ 抽 {len(grp)}")

    rows = []
    for e in picked:
        t = e["t"]
        i0 = max(0, int((t - CLIP_PRE) * sr))
        i1 = min(len(raw), int((t + CLIP_POST) * sr))
        tag = "self" if e.get("self") else "unk"
        name = f"{e['type']}_{int(t):04d}s_{tag}.wav"
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
                     "label": "", "note": ""})

    with open(out / "clips.csv", "w", newline="", encoding="utf-8-sig") as fp:
        cols = list(rows[0].keys()) if rows else \
            ["clip", "t", "type", "auto_self", "self_reason", "corr", "rms",
             "label", "note"]
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    (out / "README.txt").write_text(
        "听声标注说明\n"
        "==========\n"
        f"clips/ 里 {len(rows)} 个短片段（事件前后各 1.2 秒），clips.csv 是清单。\n\n"
        "怎么标：用 Excel/WPS 打开 clips.csv，在 label 列填你听出来的结论，note 列随手写。\n"
        "建议标签词汇（自由发挥也行，我能看懂就行）：\n"
        "  脚步:  enemy_footstep(真人) / bot_footstep(人机) / self_footstep(自己)\n"
        "  枪声:  enemy_gunshot / bot_gunshot / self_gunshot\n"
        "  其他:  glass / reload / heal / 环境音 / 听不出\n"
        "标完把 csv 发我，我对齐进数据集（学习回路的真材实料）。\n"
        "auto_self/auto_corr 列是引擎自己的判断，别受它影响，凭耳朵。\n",
        encoding="utf-8")
    print(f"\n资料包: {out}  （{len(rows)} clips + clips.csv + README.txt）")


if __name__ == "__main__":
    main()

"""姿态模板训练：用 pose_events.jsonl 的带姿态脚步窗，算每类嵌入中心。

与 rt_suppress 完全同一条特征路径（learn_embed.seg_patch + embed_v0.pt），
窗口取 [t-PRE, t+EARLY_CUT] 补零到 0.5s——和线上截断推理一致。
产出 dsp/models/pose_protos.json：
  {"walk": {"v": [..], "n": 120}, "crouch": {...}, ...}
只保留样本数 >= --min-n 的姿态；同时打印类间中心距，
中心距太近的类（比如 walk vs run）说明声学上暂时分不开，
靠键盘授权区分即可，不影响线上用法。

2026-10-01：首版素材 150719+150622（靶场自录，全是自己脚步，
self 由键盘闩锁背书），跑/趴待用户重录后追加。
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import learn_embed as le
from self_suppress import load_model_protos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", nargs="+", required=True,
                    help="会话目录名（需已有 pose_events.jsonl）")
    ap.add_argument("--out", default=None,
                    help="输出路径，默认 dsp/models/pose_protos.json")
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--early-cut", type=float, default=0.12,
                    help="与 rt_suppress.EARLY_CUT 一致")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_path = Path(args.out) if args.out else root / "dsp" / "models" / "pose_protos.json"
    model, mu, sd, ps, pf, n_train = load_model_protos()

    acc = defaultdict(list)
    for sid in args.session:
        ep = root / "sessions" / sid / "pose_events.jsonl"
        if not ep.exists():
            print(f"{sid}: 无 pose_events.jsonl，跳过")
            continue
        mono = le.load_mono(root / "sessions" / sid) \
            if (root / "sessions" / sid / "labeling").exists() else None
        if mono is None:
            (root / "sessions" / sid / "labeling").mkdir(exist_ok=True)
            mono = le.load_mono(root / "sessions" / sid)
        n = 0
        for l in open(ep, encoding="utf-8"):
            e = json.loads(l)
            pose, t = e.get("pose"), e["t"]
            if not pose:
                continue
            i0 = int((t - le.PRE) * le.TARGET_SR)
            i1 = int((t + args.early_cut) * le.TARGET_SR)
            if i0 < 0 or i1 > len(mono):
                continue
            seg = np.pad(mono[i0:i1], (0, max(0, le.SEG_LEN - (i1 - i0))))
            acc[pose].append(seg.astype(np.float32))
            n += 1
        print(f"{sid}: {n} 条姿态脚步")

    import torch
    protos = {}
    for pose, segs in sorted(acc.items()):
        if len(segs) < args.min_n:
            print(f"{pose}: 仅 {len(segs)} 条 < min-n={args.min_n}，跳过")
            continue
        embs = []
        for i in range(0, len(segs), 64):
            xb = np.stack([le.seg_patch(s) for s in segs[i:i + 64]])
            xb = torch.tensor((xb - mu) / sd)[:, None]
            with torch.no_grad():
                e = torch.nn.functional.normalize(model(xb), dim=1)
            embs.append(e.numpy())
        embs = np.concatenate(embs)
        c = embs.mean(0)
        c /= np.linalg.norm(c) + 1e-12
        protos[pose] = {"v": [round(float(v), 6) for v in c],
                        "n": int(len(segs))}
        print(f"{pose}: n={len(segs)}  内聚(平均cos)="
              f"{float(np.mean(embs @ c)):.3f}")

    poses = list(protos)
    print("\n类间中心余弦距（越大分得越开）：")
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            a = np.array(protos[poses[i]]["v"])
            b = np.array(protos[poses[j]]["v"])
            print(f"  {poses[i]:8s} vs {poses[j]:8s}: d={1 - float(a @ b):.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"sessions": args.session, "min_n": args.min_n},
               "protos": protos}, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()

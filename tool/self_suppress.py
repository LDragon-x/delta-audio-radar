# -*- coding: utf-8 -*-
"""自我脚步消噪 v0.1（离线版）：embed 模型判定 self/foe，self 事件窗压增益。

原理：对自动检测到的每个脚步事件提 embedding，与训练中「自己脚步 / 他人脚步」
两个原型比余弦距离，分差 > tau 判为自己 → 该事件窗（前 0.15s ~ 后 0.35s）
整体压 gain_db（余弦斜坡进出，避免爆音）。 overlapped 窗口取最小增益。

注意（诚实声明）：这是增益门控不是频谱减除——self 窗内若有敌人声音会被一起压。
交战段是否该禁用门控，留给 v0.2 按枪声闩锁决定。

用法：
  python tool/self_suppress.py --session sessions/20261001_125839 [--gain-db 15] [--tau 0.0]

输出（sessions/<ts>/suppress/）：
  cleaned.wav        消噪后整局（16k mono）
  report.json        抑制统计（对照 auto_events 的 self 标签算抑制率/误伤率）
  ab_<i>_orig.wav / ab_<i>_clean.wav   A/B 对比段（挑 self 最密集处）
"""
import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tool"))
import learn_embed as le  # noqa: E402

RAMP = 0.08  # 增益斜坡秒数


def write_wav(path, sig, sr=16000):
    sig = np.clip(sig, -1, 1)
    pcm = (sig * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--gain-db", type=float, default=15.0)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--ab-n", type=int, default=3)
    ap.add_argument("--no-pose", action="store_true",
                    help="禁用姿态门控（对比用，默认启用：pose.jsonl 存在时生效）")
    args = ap.parse_args()
    sess = Path(args.session)
    out_dir = sess / "suppress"
    out_dir.mkdir(exist_ok=True)

    model, mu, sd, ps, pf, n_train = load_model_protos()
    mono = le.load_mono(sess)
    sr = le.TARGET_SR
    ev = [json.loads(l) for l in (sess / "auto_events.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    fs = [e for e in ev if e["type"] == "footstep"]

    # 逐个事件打分
    segs, idx = [], []
    for i, e in enumerate(fs):
        i0 = max(0, int((e["t"] - le.PRE) * sr))
        seg = mono[i0:i0 + le.SEG_LEN]
        if len(seg) < le.SEG_LEN:
            continue
        segs.append(seg)
        idx.append(i)
    rows = [(s, None, None, None) for s in segs]
    emb = le.embed_rows(model, mu, sd, rows)
    for j, e_i in enumerate(idx):
        fs[e_i]["score"] = float(emb[j] @ ps - emb[j] @ pf)

    # 姿态门控（与 rt_suppress.py 同款规则，2026-10-01）：
    # pose.jsonl 存在时，按键盘姿态换专属模板 + 授权加分 + 站立保护。
    import pose_label as pl
    pose_protos = {}
    pp_path = ROOT / "dsp" / "models" / "pose_protos.json"
    if pp_path.exists():
        pose_protos = {p: np.array(d["v"], dtype=np.float32)
                       for p, d in json.load(open(pp_path, encoding="utf-8"))["protos"].items()}
    tl = []
    pf_path = sess / "pose.jsonl"
    if not args.no_pose and pf_path.exists():
        tl = [(r["t"], r["pose"]) for r in
              (json.loads(l) for l in open(pf_path, encoding="utf-8"))]
        print(f"姿态门控: 开（{len(tl)} 次翻转, 模板 {list(pose_protos)}）")
    for j, e_i in enumerate(idx):
        if not tl:
            continue
        pose = pl.pose_at(tl, fs[e_i]["t"])
        fs[e_i]["pose"] = pose
        if pose in pose_protos:
            s_pose = float(emb[j] @ pose_protos[pose] - emb[j] @ pf)
            fs[e_i]["score"] = max(fs[e_i]["score"], s_pose)
        if pose in ("walk", "run", "jump", "crouch", "crawl"):
            fs[e_i]["score"] += 0.10
        elif pose is None:
            fs[e_i]["score"] -= 0.35

    # 增益包络 v0.2：置信度软增益（sigmoid 温度 0.1）+ 枪声闩锁
    # （开火后 2.5s 内深抑制关闭，最低只压 -6dB——交战段保住敌人声音）
    g_min = 10.0 ** (-args.gain_db / 20.0)
    g_floor = 10.0 ** (-6.0 / 20.0)
    env = np.ones(len(mono), dtype=np.float32)
    n_ramp = int(RAMP * sr)
    for e in fs:
        s = e.get("score")
        if s is None:
            continue
        g = 1.0 - (1.0 - g_min) * float(1.0 / (1.0 + np.exp(-(s - args.tau) / 0.1)))
        if g > 0.995:
            continue
        c = int(e["t"] * sr)
        i0 = max(0, c - int(le.PRE * sr))
        i1 = min(len(mono), c + int(le.POST * sr))
        L = i1 - i0
        w = np.ones(L, dtype=np.float32)
        nr = min(n_ramp, L // 2)
        if nr > 0:
            r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, nr))
            w[:nr] = 1 - (1 - g) * r
            w[-nr:] = 1 - (1 - g) * r[::-1]
        w[nr:L - nr] = g
        env[i0:i1] = np.minimum(env[i0:i1], w)
    latch = np.zeros(len(mono), dtype=bool)
    for e in ev:
        if e["type"] != "gunshot":
            continue
        c = int(e["t"] * sr)
        latch[max(0, c - int(1.0 * sr)):min(len(mono), c + int(2.5 * sr))] = True
    env = np.where(latch, np.maximum(env, g_floor), env)
    cleaned = mono * env
    write_wav(out_dir / "cleaned.wav", cleaned)

    # 对照 auto_events 的 self 标签（键盘闩锁）算抑制统计
    def deep(e):  # 深度抑制判定：事件中心增益 ≤ 1.5×g_min
        return env[int(e["t"] * sr)] <= g_min * 1.5
    def att_db(e):  # 事件中心实际压了多少 dB
        return -20.0 * float(np.log10(env[int(e["t"] * sr)] + 1e-9))
    self_ev = [e for e in fs if e.get("self")]
    oth_ev = [e for e in fs if not e.get("self")]
    n_deep = sum(1 for e in fs if e.get("score") is not None
                 and 1.0 - (1.0 - g_min) / (1.0 + np.exp(-(e["score"] - args.tau) / 0.1))
                 <= g_min * 1.5)
    rep = {
        "session": sess.name, "version": "v0.2-soft+gunshot-latch",
        "tau": args.tau, "gain_db": args.gain_db,
        "n_events": len(fs), "n_deep": n_deep,
        "self_total": len(self_ev),
        "self_deep_rate": sum(deep(e) for e in self_ev) / max(len(self_ev), 1),
        "self_median_att_db": float(np.median([att_db(e) for e in self_ev]))
        if self_ev else 0.0,
        "other_total": len(oth_ev),
        "other_deep_rate": sum(deep(e) for e in oth_ev) / max(len(oth_ev), 1),
        "env_cover": float((env < 0.99).mean()),
        "n_train": n_train,
    }
    (out_dir / "report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    # A/B 段：从 self=true 且 score 高的事件里挑不重叠的 4s 段
    cand = sorted([e for e in self_ev if e.get("score", 9) > args.tau],
                  key=lambda e: -e["feat"]["rms"])
    picks, used = [], []
    for e in cand:
        c = e["t"]
        if any(abs(c - u) < 5 for u in used):
            continue
        used.append(c)
        picks.append(c)
        if len(picks) >= args.ab_n:
            break
    for i, c in enumerate(picks):
        i0, i1 = int((c - 1.5) * sr), int((c + 2.5) * sr)
        i0, i1 = max(0, i0), min(len(mono), i1)
        write_wav(out_dir / f"ab_{i}_orig.wav", mono[i0:i1])
        write_wav(out_dir / f"ab_{i}_clean.wav", cleaned[i0:i1])

    print(f"事件 {len(fs)}，深度抑制 {rep['n_deep']}（{rep['n_deep']/max(len(fs),1):.0%}）")
    print(f"self 深度抑制率 {rep['self_deep_rate']:.1%}（{len(self_ev)} 条，中位压 "
          f"{rep['self_median_att_db']:.1f}dB），"
          f"other 深度误伤率 {rep['other_deep_rate']:.1%}（{len(oth_ev)} 条）")
    print(f"包络覆盖 {(env < 0.99).mean():.2%} 时长；A/B 段 {len(picks)} 组")
    print(f"→ {out_dir}")


def load_model_protos():
    ckpt = torch.load(ROOT / "dsp" / "models" / "embed_v0.pt", weights_only=False)
    model = le.EmbedCNN()
    model.load_state_dict(ckpt["state"])
    model.eval()
    data = le.collect()
    rows = [r for v in data.values() for r in v]
    emb = le.embed_rows(model, ckpt["mu"], ckpt["sd"], rows)
    ps = emb[[i for i, r in enumerate(rows) if r[3] == "self"]]
    pf = emb[[i for i, r in enumerate(rows) if r[3] == "foe"]]
    assert len(ps) and len(pf), "原型所需标注缺失"
    ps = ps.mean(0)
    pf = pf.mean(0)
    ps /= np.linalg.norm(ps) + 1e-9
    pf /= np.linalg.norm(pf) + 1e-9
    return model, ckpt["mu"], ckpt["sd"], ps, pf, len(rows)


if __name__ == "__main__":
    main()

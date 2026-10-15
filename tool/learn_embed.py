# -*- coding: utf-8 -*-
"""embedding 判定器 v0.2：小 CNN 学脚步 embedding，按局留出评估。

v0.1 教训（JOURNAL 留痕）：训练集内 acc 86%，但留出局上原型打分检出 0%——
embedding 的局间差异（底噪/响度/混响指纹）远大于类间差异，
跨局时类结构被冲掉。典型 domain shift。

v0.2 针对性增强（训练时波形域做，不是语谱图遮遮掩掩）：
  - 增益抖动 ±6dB         → 打掉响度/采集增益指纹
  - 背景混洗 -15~-40dB    → 从其他局随机抓背景段叠加，
                             强迫模型对「底噪是谁」不变，只学事件本体
  - 保留 SpecAugment 时频掩码

类别（单标签优先级近似）：self_footstep > footstep(真人) > bot_footstep > other
评估：按局留出，5-NN（与烘焙赛同口径）+ 原型打分（You2023 口径）双报。

用法：
  python tool/learn_embed.py            # 8 折留出评估 + 存最终模型
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tool"))
from label_map import class_of  # noqa: E402
from dsp.logmel import logmel_patch, stft_power, mel_bank  # noqa: E402

torch.set_num_threads(1)  # CPU 多线程带来 run-to-run 不确定性，小数据下结果会飘
SEED = 7
TARGET_SR = 16000
PRE, POST = 0.15, 0.35          # 事件窗，与 logmel_patch 默认一致
SEG_LEN = int((PRE + POST) * TARGET_SR)
CLASSES = ["self_footstep", "footstep", "bot_footstep", "other"]


def load_mono(session_dir):
    cache = Path(session_dir) / "labeling" / "_mono16k.npy"
    if cache.exists():
        return np.load(cache)
    import wave
    with wave.open(str(Path(session_dir) / "session.wav"), "rb") as w:
        sr, ch, sw, n = (w.getframerate(), w.getnchannels(),
                         w.getsampwidth(), w.getnframes())
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch).mean(axis=1)
    a = a[::(sr // TARGET_SR)].astype(np.float32) / 32768.0
    np.save(cache, a)
    return a


def primary_class(cls_set):
    if "self_footstep" in cls_set:
        return "self_footstep"
    if "footstep" in cls_set:
        return "footstep"
    if "bot_footstep" in cls_set:
        return "bot_footstep"
    return "other"


_MEL = {}


def seg_patch(seg, n_fft=512, hop=128, n_mels=64, fmin=40.0):
    """0.5s 波形段 → log-mel patch (64, 59)。与 logmel_patch 同参数。"""
    key = n_fft
    if key not in _MEL:
        _MEL[key] = mel_bank(TARGET_SR, n_fft, n_mels, fmin, None)
    if len(seg) < SEG_LEN:
        seg = np.pad(seg, (0, SEG_LEN - len(seg)))
    power = stft_power(seg, n_fft, hop)
    mel = power @ _MEL[key].T
    return np.log(mel + 1e-10).T.astype(np.float32)


def collect():
    """→ {session: [(seg, y, task_a, task_b)]}，seg 为事件窗波形。"""
    out = {}
    for d in sorted(glob.glob(str(ROOT / "sessions" / "2*"))):
        d = Path(d)
        cf, uf = d / "labeling" / "clips.csv", d / "labeling" / "user_labels.csv"
        if not (cf.exists() and uf.exists()):
            continue
        tmap = {r["clip"]: float(r["t"])
                for r in csv.DictReader(open(cf, encoding="utf-8-sig"))}
        mono = None
        rows = []
        for r in csv.DictReader(open(uf, encoding="utf-8-sig")):
            label = (r.get("label") or "").strip()
            t = tmap.get(r["clip"])
            if not label or t is None:
                continue
            cls_set = {class_of(l.strip()) for l in label.split(";")}
            cls_set.discard(None)
            y = primary_class(cls_set)
            if y is None:
                continue
            has_step = bool({"self_footstep", "footstep", "bot_footstep"}
                            & cls_set)
            task_a = ("bot" if "bot_footstep" in cls_set else
                      "hum" if "footstep" in cls_set else None)
            task_b = ("self" if "self_footstep" in cls_set else
                      "foe" if has_step else None)
            if mono is None:
                mono = load_mono(d)
            i0 = max(0, int((t - PRE) * TARGET_SR))
            seg = mono[i0:i0 + SEG_LEN]
            rows.append((seg.astype(np.float32), y, task_a, task_b))
        if rows:
            out[d.name] = rows
    return out


class EmbedCNN(nn.Module):
    """4 层 CNN（Yang2022 量级）：1×64×59 → embedding 32 维。"""

    def __init__(self, emb=32, n_cls=4):
        super().__init__()
        ch = [1, 16, 32, 64, 64]
        blocks = []
        for i in range(4):
            blocks += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1),
                       nn.BatchNorm2d(ch[i + 1]), nn.ReLU(),
                       nn.MaxPool2d(2)]
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(64 * 4 * 3, 128), nn.ReLU(),
                                  nn.Linear(128, emb))
        self.clf = nn.Linear(emb, n_cls)

    def forward(self, x):
        return self.head(self.conv(x).flatten(1))

    def forward_class(self, x):
        return self.clf(self.forward(x))


def spec_augment(x, n_time=2, n_freq=2, t_w=10, f_w=8):
    """时频掩码，x (C,64,T)，掩区填 0（已归一化，0≈全局均值）。"""
    x = x.clone()
    _, Fb, T = x.shape
    for _ in range(n_time):
        w = int(np.random.randint(1, t_w + 1))
        s = int(np.random.randint(0, max(T - w, 1)))
        x[:, :, s:s + w] = 0.0
    for _ in range(n_freq):
        w = int(np.random.randint(1, f_w + 1))
        s = int(np.random.randint(0, max(Fb - w, 1)))
        x[:, s:s + w, :] = 0.0
    return x


def gain_jitter(seg, db=6.0):
    g = 10.0 ** (np.random.uniform(-db, db) / 20.0)
    return seg * g


def bg_mix(seg, bg_pool, snr_range=(15.0, 40.0)):
    """从背景池随机抓一段，按随机 SNR 叠加。bg_pool: 各局 mono 列表。"""
    if not bg_pool or np.random.rand() > 0.7:
        return seg
    bg = bg_pool[np.random.randint(len(bg_pool))]
    if len(bg) < SEG_LEN:
        return seg
    s = np.random.randint(0, len(bg) - SEG_LEN)
    b = bg[s:s + SEG_LEN]
    p_s = float((seg ** 2).mean()) + 1e-12
    p_b = float((b ** 2).mean()) + 1e-12
    snr = np.random.uniform(*snr_range)
    scale = np.sqrt(p_s / p_b) * 10.0 ** (-snr / 20.0)
    return seg + b * scale


def trunc_aug(seg):
    """截断增强（实时低延迟必需）：以 50% 概率只留前 30~100%，其余补零。
    让模型学会"看半个窗口就判敌我"，分类等待从 0.35s 砍到 0.12s。"""
    if np.random.rand() < 0.5:
        k = np.random.uniform(0.3, 1.0)
        n = int(len(seg) * k)
        return np.pad(seg[:n], (0, len(seg) - n))
    return seg


def augment_batch(segs, bg_pool):
    patches = [seg_patch(trunc_aug(bg_mix(gain_jitter(s), bg_pool)))
               for s in segs]
    return np.stack(patches)


def train_fold(train_by_sess, epochs, seed=SEED):
    """train_by_sess: {session: rows}（折内训练局）。返回 model/mu/sd。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rows = [r for v in train_by_sess.values() for r in v]
    bg_pool = []
    for s, v in train_by_sess.items():
        mono = load_mono(ROOT / "sessions" / s)
        # 背景池只收训练局波形，叠加时随机开窗
        bg_pool.append(mono)
    # 归一化统计：干净 patch 上算
    X = np.stack([seg_patch(r[0]) for r in rows])
    mu, sd = float(X.mean()), float(X.std() + 1e-6)
    y = np.array([CLASSES.index(r[1]) for r in rows])
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    weights = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    model = EmbedCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(weights))
    n = len(rows)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(n)
        for i in range(0, n, 32):
            b = idx[i:i + 32]
            Xb = augment_batch([rows[j][0] for j in b], bg_pool)
            xb = torch.tensor((Xb - mu) / sd)[:, None]
            xb = torch.stack([spec_augment(v) for v in xb])
            opt.zero_grad()
            loss = lossf(model.forward_class(xb), torch.tensor(y[b]))
            loss.backward()
            opt.step()
    model.eval()
    return model, mu, sd


@torch.no_grad()
def embed_rows(model, mu, sd, rows):
    X = np.stack([seg_patch(r[0]) for r in rows])
    Xn = (X - mu) / sd
    embs = []
    for i in range(0, len(Xn), 64):
        xb = torch.tensor(Xn[i:i + 64])[:, None]
        e = torch.nn.functional.normalize(model(xb), dim=1)
        embs.append(e.numpy())
    return np.concatenate(embs)


def ti_adapt(pos_c, neg_c, q_emb, epochs=30, lr=1e-3):
    """转导推理（Yang2022）：W 从训练原型初始化，只用留出局无标注 query 的
    互信息损失更新 W（边缘熵 - 条件熵），让分类边界避开 query 稠密区。"""
    W = torch.tensor(np.stack([pos_c, neg_c]), dtype=torch.float32,
                     requires_grad=True)
    opt = torch.optim.Adam([W], lr=lr)
    Q = torch.tensor(q_emb, dtype=torch.float32)
    for _ in range(epochs):
        p = torch.softmax(Q @ W.T, dim=1)
        marg = p.mean(0)
        mi = -(marg * torch.log(marg + 1e-9)).sum() \
             + (p * torch.log(p + 1e-9)).sum() / len(Q)
        opt.zero_grad()
        (-mi).backward()
        opt.step()
    wd = W.detach().numpy()
    return wd[0], wd[1]


def query_embeddings(held):
    """留出局无标注脚步事件（auto_events type=footstep）的 embedding。"""
    ev_path = ROOT / "sessions" / held / "auto_events.jsonl"
    if not ev_path.exists():
        return None
    ev = [json.loads(l) for l in ev_path.read_text(encoding="utf-8").splitlines()
          if l.strip()]
    ts = [e["t"] for e in ev if e.get("type") == "footstep"][:200]
    if not ts:
        return None
    mono = load_mono(ROOT / "sessions" / held)
    rows = [(mono[max(0, int((t - PRE) * TARGET_SR)):][:SEG_LEN], "q", None, None)
            for t in ts]
    return rows


def knn_eval(train_pairs, test_rows):
    """与 feat_eval 同口径：5-NN 多数表决（pos 票≥3 判 pos）。"""
    tr = [(e, lab) for e, _, lab in train_pairs if lab]
    tp = fp = fn = tn = 0
    for e, r, lab in test_rows:
        if not lab:
            continue
        near = sorted(tr, key=lambda ts: float(np.linalg.norm(ts[0] - e)))[:5]
        pred_pos = sum(1 for _, l in near if l == "pos") >= 3
        if pred_pos and lab == "pos":
            tp += 1
        elif pred_pos:
            fp += 1
        elif lab == "pos":
            fn += 1
        else:
            tn += 1
    return tp / max(tp + fn, 1), fp / max(fp + tn, 1)


def proto_eval(train_pairs, test_rows):
    """原型打分（You2023 口径）：cos 距离二选一，类别天然均衡。"""
    pos = np.mean([e for e, _, lab in train_pairs if lab == "pos"], axis=0)
    neg = np.mean([e for e, _, lab in train_pairs if lab == "neg"], axis=0)
    tp = fp = fn = tn = 0
    for e, r, lab in test_rows:
        if not lab:
            continue
        pred_pos = float(e @ pos) > float(e @ neg)
        if pred_pos and lab == "pos":
            tp += 1
        elif pred_pos:
            fp += 1
        elif lab == "pos":
            fn += 1
        else:
            tn += 1
    return tp / max(tp + fn, 1), fp / max(fp + tn, 1)


def _proto_with_centroids(pos_vec, neg_vec, test_rows):
    """给定已适配/固定的正负质心向量，对测试行做原型打分（TI 用）。"""
    tp = fp = fn = tn = 0
    for e, r, lab in test_rows:
        if not lab:
            continue
        pred_pos = float(e @ pos_vec) > float(e @ neg_vec)
        if pred_pos and lab == "pos":
            tp += 1
        elif pred_pos:
            fp += 1
        elif lab == "pos":
            fn += 1
        else:
            tn += 1
    return tp / max(tp + fn, 1), fp / max(fp + tn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(ROOT / "dsp" / "models" / "embed_v0.pt"))
    ap.add_argument("--results-jsonl", default=None,
                    help="指定则把每折结果追加到该文件（多种子跑法用）")
    ap.add_argument("--no-final-model", action="store_true")
    ap.add_argument("--ti", action="store_true",
                    help="每折用留出局无标注脚步事件做转导推理适配原型（Yang2022）")
    args = ap.parse_args()

    data = collect()
    total = sum(len(v) for v in data.values())
    dist = " ".join(f"{c}={sum(1 for v in data.values() for r in v if r[1] == c)}"
                    for c in CLASSES)
    print(f"{len(data)} 局 {total} 条标注段；{dist}")

    folds = sorted(data)
    res_a, res_b = [], []
    rj = open(args.results_jsonl, "a", encoding="utf-8") if args.results_jsonl \
        else None
    for held in folds:
        train_by_sess = {s: v for s, v in data.items() if s != held}
        model, mu, sd = train_fold(train_by_sess, args.epochs, seed=args.seed)
        train_rows = [r for v in train_by_sess.values() for r in v]
        tr_emb = embed_rows(model, mu, sd, train_rows)
        te_rows = data[held]
        te_emb = embed_rows(model, mu, sd, te_rows)
        tr_a = [(tr_emb[i], train_rows[i],
                 "pos" if train_rows[i][2] == "bot" else
                 "neg" if train_rows[i][2] == "hum" else None)
                for i in range(len(train_rows))]
        tr_b = [(tr_emb[i], train_rows[i],
                 "pos" if train_rows[i][3] == "self" else
                 "neg" if train_rows[i][3] == "foe" else None)
                for i in range(len(train_rows))]
        te_a = [(te_emb[i], te_rows[i],
                 "pos" if te_rows[i][2] == "bot" else
                 "neg" if te_rows[i][2] == "hum" else None) for i in
                range(len(te_rows))]
        te_b = [(te_emb[i], te_rows[i],
                 "pos" if te_rows[i][3] == "self" else
                 "neg" if te_rows[i][3] == "foe" else None) for i in
                range(len(te_rows))]
        da, fa = knn_eval(tr_a, te_a)
        db, fb = knn_eval(tr_b, te_b)
        pa, qa = proto_eval(tr_a, te_a)
        pb, qb = proto_eval(tr_b, te_b)
        res_a.append((da, fa, pa, qa))
        res_b.append((db, fb, pb, qb))
        ti_line = ""
        if args.ti:
            q_rows = query_embeddings(held)
            if q_rows:
                q_emb = embed_rows(model, mu, sd, q_rows)
                pa2, qa2 = _proto_with_centroids(
                    *ti_adapt(np.mean([e for e, _, l in tr_a if l == "pos"], axis=0),
                              np.mean([e for e, _, l in tr_a if l == "neg"], axis=0),
                              q_emb), te_a)
                pb2, qb2 = _proto_with_centroids(
                    *ti_adapt(np.mean([e for e, _, l in tr_b if l == "pos"], axis=0),
                              np.mean([e for e, _, l in tr_b if l == "neg"], axis=0),
                              q_emb), te_b)
                ti_line = f"  TI后 任务A {pa2:.0%}/{qa2:.0%} 任务B {pb2:.0%}/{qb2:.0%}"
        if rj:
            rec = {"seed": args.seed, "held": held,
                   "A_5nn": [da, fa], "A_proto": [pa, qa],
                   "B_5nn": [db, fb], "B_proto": [pb, qb]}
            if ti_line:
                rec["A_proto_ti"] = [pa2, qa2]
                rec["B_proto_ti"] = [pb2, qb2]
            rj.write(json.dumps(rec) + "\n")
            rj.flush()
        print(f"留出 {held}: 任务A 5NN {da:.0%}/{fa:.0%} 原型 {pa:.0%}/{qa:.0%}"
              f"  任务B 5NN {db:.0%}/{fb:.0%} 原型 {pb:.0%}/{qb:.0%}{ti_line}")
    if rj:
        rj.close()

    ma = np.mean([(r[0], r[1]) for r in res_a], axis=0)
    mp = np.mean([(r[2], r[3]) for r in res_a], axis=0)
    mb = np.mean([(r[0], r[1]) for r in res_b], axis=0)
    mq = np.mean([(r[2], r[3]) for r in res_b], axis=0)
    print(f"\n均值(5NN)  任务A 检出 {ma[0]:.0%} 误伤 {ma[1]:.0%}"
          f"（烘焙赛 log-mel all: 39%/88%）")
    print(f"均值(5NN)  任务B 检出 {mb[0]:.0%} 误伤 {mb[1]:.0%}"
          f"（烘焙赛 log-mel all: 81%/48%）")
    print(f"均值(原型)任务A 检出 {mp[0]:.0%} 误伤 {mp[1]:.0%}")
    print(f"均值(原型)任务B 检出 {mq[0]:.0%} 误伤 {mq[1]:.0%}")

    if args.no_final_model:
        return
    model, mu, sd = train_fold(data, args.epochs, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd,
                "classes": CLASSES, "version": "v0.2"}, out)
    meta = {"model": str(out), "classes": CLASSES, "mu": mu, "sd": sd,
            "n_train": total, "version": "v0.2",
            "loso_5nn": {"taskA": [list(map(float, (r[0], r[1]))) for r in res_a],
                         "taskB": [list(map(float, (r[0], r[1]))) for r in res_b]},
            "loso_proto": {
                "taskA": [list(map(float, (r[2], r[3]))) for r in res_a],
                "taskB": [list(map(float, (r[2], r[3]))) for r in res_b]}}
    (out.parent / "embed_v0_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n最终模型 → {out}（{total} 条全量训练）")


def embed_event(mono, t, model_dir=None):
    """推理入口：单事件 → L2 归一 embedding（需先训出 embed_v0.pt）。"""
    model_dir = Path(model_dir or ROOT / "dsp" / "models")
    ckpt = torch.load(model_dir / "embed_v0.pt", weights_only=False)
    model = EmbedCNN()
    model.load_state_dict(ckpt["state"])
    model.eval()
    i0 = max(0, int((t - PRE) * TARGET_SR))
    patch = seg_patch(mono[i0:i0 + SEG_LEN])
    x = torch.tensor((patch - ckpt["mu"]) / ckpt["sd"])[None, None]
    with torch.no_grad():
        e = torch.nn.functional.normalize(model(x), dim=1)
    return e.numpy()[0]


if __name__ == "__main__":
    main()

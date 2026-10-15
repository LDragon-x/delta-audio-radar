"""内置学习回路 v0.1：用采集器攒的人工标签训练事件分类器。

学习闭环（每局打完自动变聪明一点）：
  采集器录局 → 你按 F1/F2/F3 标标签 → learn.py 时间对齐(±0.5s)拿自动事件的
  特征配人工标签 → k-NN 训练 + 留出法验证 → 模型存 samples/model/knn.json
  → 下局 BotFingerprint / 分类器加载模型替代手写规则

用法：
  python tool/learn.py            # 训练并出验证报告
  python tool/learn.py --stats    # 只看数据集统计
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_map import class_of  # 标注值（新/旧/随手写）→ 规范类统一口径

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "sessions"
MODEL_PATH = ROOT / "samples" / "model" / "knn.json"

# 特征向量键（与 collector 保存的 feat 一致）+ 是否取 log
FEAT_KEYS = ["low", "mid", "high", "low_ratio", "high_ratio", "corr", "low_flat", "rms"]
LOG_KEYS = {"low", "mid", "high", "rms"}
MATCH_WINDOW = 0.5  # 人工标签与自动事件的时间对齐窗口
# 可训练类别：脚步/枪声三类 + 用户实战中发明的新声音类（播报/搜物资/
# 人机叫声/近战/技能/救援——既是独立类别，也是将来虚拟场景的干扰声源）
TRAINABLE = {"footstep", "bot_footstep", "gunshot",
             "broadcast", "loot", "bot_vocal", "melee", "skill", "rescue"}


def feat_vec(f):
    return [float(np.log(f[k] + 1e-9)) if k in LOG_KEYS else float(f[k])
            for k in FEAT_KEYS]


def load_dataset():
    """返回 [(vec, label)]：人工标签 ±0.5s 内最近的自动事件特征。

    两个来源：labels.jsonl（打标热键/虚拟会话真值）+ 边听边标的
    user_labels.csv（经 clips.csv 取时刻；分号多标签展开成多条样本）。
    """
    samples = []
    stats = {}
    for d in sorted(SESSIONS.glob("*")):
        af = d / "auto_events.jsonl"
        if not af.exists():
            continue
        autos = [json.loads(x) for x in af.read_text(encoding="utf-8").splitlines()
                 if x.strip()]

        def absorb(t, name):
            cls = class_of(name)
            if cls not in TRAINABLE:
                return
            near = [a for a in autos
                    if abs(a["t"] - t) <= MATCH_WINDOW and "feat" in a]
            if not near:
                return
            a = min(near, key=lambda x: abs(x["t"] - t))
            samples.append((feat_vec(a["feat"]), cls))
            stats[cls] = stats.get(cls, 0) + 1

        lf = d / "labels.jsonl"
        if lf.exists():
            for line in lf.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    lab = json.loads(line)
                    absorb(lab["t"], lab["label"])
        # 边听边标：clip 时刻表 + 多标签（; 分隔）；同 clip 同规范类去重
        # （"enemy" 和 "enemy:脚步" 都映射 footstep，只留一条样本）
        cf, uf = d / "labeling" / "clips.csv", d / "labeling" / "user_labels.csv"
        if cf.exists() and uf.exists():
            tmap = {}
            with open(cf, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    tmap[r["clip"]] = float(r["t"])
            seen = set()
            with open(uf, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    t = tmap.get(r["clip"])
                    if t is None or not r.get("label"):
                        continue
                    for lab in r["label"].split(";"):
                        cls = class_of(lab.strip())
                        if cls not in TRAINABLE or (r["clip"], cls) in seen:
                            continue
                        seen.add((r["clip"], cls))
                        absorb(t, lab.strip())
    return samples, stats


def knn_train(samples, k=5):
    """留出法 70/30 验证 + 全量训练。返回 (report, model)。

    数据集类别极不平衡（人机脚步 300+ vs 稀有类个位数），两处对策：
    训练集每类封顶 max(40, 2×中位类量) 随机子采样；投票按 1/距离 加权，
    避免多数类吞掉近邻表决。模型仍带 training_stats 供上游判断可信度。
    """
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(samples))
    n_test = max(1, int(len(idx) * 0.3))
    test, train = idx[:n_test], idx[n_test:]

    def cap_balance(idxs):
        from collections import Counter
        counts = Counter(samples[i][1] for i in idxs)
        cap = max(40, 2 * int(np.median(list(counts.values()))))
        by_cls = {}
        for i in idxs:
            by_cls.setdefault(samples[i][1], []).append(i)
        out = []
        for cls, ii in by_cls.items():
            if len(ii) > cap:
                ii = rng.choice(ii, cap, replace=False).tolist()
            out.extend(ii)
        return out

    train = cap_balance(train)
    X = np.array([samples[i][0] for i in train])
    y = [samples[i][1] for i in train]
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9

    def predict(vecs):
        out = []
        for v in vecs:
            d = np.linalg.norm((X - (np.array(v) - mu) / sd), axis=1)
            kk = np.argsort(d)[:k]
            votes = {}
            for j in kk:
                votes[y[j]] = votes.get(y[j], 0) + 1.0 / (d[j] + 1e-6)
            out.append(max(votes, key=votes.get))
        return out

    report = {}
    if len(test):
        Xt = [samples[i][0] for i in test]
        yt = [samples[i][1] for i in test]
        pt = predict(Xt)
        for cls in sorted(set(yt) | set(pt)):
            tp = sum(1 for a, b in zip(yt, pt) if a == cls and b == cls)
            fp = sum(1 for a, b in zip(yt, pt) if a != cls and b == cls)
            fn = sum(1 for a, b in zip(yt, pt) if a == cls and b != cls)
            report[cls] = {"precision": round(tp / max(1, tp + fp), 2),
                           "recall": round(tp / max(1, tp + fn), 2),
                           "support_test": sum(1 for a in yt if a == cls)}
    model = {"k": k, "feats": FEAT_KEYS, "log_keys": sorted(LOG_KEYS),
             "mu": mu.tolist(), "sd": sd.tolist(),
             "X": X.tolist(), "y": y,
             "training_stats": {c: y.count(c) for c in sorted(set(y))}}
    return report, model


def replay_eval(model):
    """拿模型给所有真实局 auto_events（脚步/人机步/枪声）打分类，
    和引擎规则分类对账——看模型在高置信时能不能信。"""
    import numpy as np
    from collections import Counter
    X = np.array(model["X"])
    y = model["y"]
    mu, sd = np.array(model["mu"]), np.array(model["sd"])

    def predict(f):
        v = np.array(feat_vec(f))
        d = np.linalg.norm(X - (v - mu) / sd, axis=1)
        kk = np.argsort(d)[: model["k"]]
        votes = {}
        for j in kk:
            votes[y[j]] = votes.get(y[j], 0) + 1.0 / (d[j] + 1e-6)
        best = max(votes, key=votes.get)
        return best, votes[best] / sum(votes.values())

    conf, conf_hi, n = Counter(), Counter(), 0
    for d in sorted(SESSIONS.glob("*")):
        af = d / "auto_events.jsonl"
        if not af.exists():
            continue
        for line in af.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            f = e.get("feat")
            if not f or e.get("type") not in ("footstep", "bot_footstep", "gunshot"):
                continue
            n += 1
            pred, confi = predict(f)
            conf[(e["type"], pred)] += 1
            if confi >= 0.9:
                conf_hi[(e["type"], pred)] += 1
    print(f"\n回放 {n} 个引擎事件（脚步/人机步/枪声）")
    print("引擎规则类 → 模型预测（全部）：")
    for (rule, pred), c in sorted(conf.items(), key=lambda x: -x[1]):
        print(f"  {rule:14} -> {pred:14} {c}")
    print("引擎规则类 → 模型预测（仅高置信≥0.9）：")
    for (rule, pred), c in sorted(conf_hi.items(), key=lambda x: -x[1]):
        print(f"  {rule:14} -> {pred:14} {c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--eval", action="store_true",
                    help="用已训模型回放全部真实局事件，和引擎规则对账")
    args = ap.parse_args()

    samples, stats = load_dataset()
    print(f"数据集：{stats}，共 {len(samples)} 条对齐样本")
    if args.eval:
        if MODEL_PATH.exists():
            replay_eval(json.loads(MODEL_PATH.read_text(encoding="utf-8")))
        else:
            print("没有已训模型——先跑 python tool/learn.py")
        return
    if args.stats or not samples:
        if not samples:
            print("没有可对齐样本——先跑几局采集器并按 F1/F2 标标签")
        return

    report, model = knn_train(samples)
    print("\n留出法验证（30% 测试集）：")
    for cls, r in sorted(report.items()):
        print(f"  {cls:<12} 精确率 {r['precision']:.2f}  召回率 {r['recall']:.2f}  "
              f"测试样本 {r['support_test']}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    print(f"\n模型已存 {MODEL_PATH}（{len(model['X'])} 条训练样本）")


if __name__ == "__main__":
    main()

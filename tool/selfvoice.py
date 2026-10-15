"""自身声纹库工具：攒"自己的声音"指纹，给防线3（dsp/selfsound.py SelfVoiceprint）用。

指纹几何与运行时完全一致（vp_window: 触发点 -50ms ~ +130ms，攻击窗 80ms）——
建库和匹配走同一条截取路径，线上线下才对得齐。

用法：
  # 从 collector 会话建库（自标标签 r/b + 规则已标 self 的自动事件）
  python tool/selfvoice.py build --session sessions/20260927_xxxxxx

  # 手动加一条（wav + 时间点 + 类别）
  python tool/selfvoice.py add --file session.wav --t 12.34 --cat self_footstep

  # 验证某个时间点是敌是我（打印到各类别最近距离）
  python tool/selfvoice.py match --file session.wav --t 56.7

  # 看库内容
  python tool/selfvoice.py list

类别约定：self_footstep / self_gunshot / self_reload / self_firemode / self_heal ...
命中阈值在 config.json self_voiceprint.thresh（起始 0.30，干净样本标定后收紧）。
"""
import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp.guntype import fingerprint, cosine_dist  # noqa: E402
from dsp.selfsound import VP_PRE, VP_POST, VP_ATTACK  # noqa: E402

DEFAULT_LIB = "samples/self/library.json"

# collector 标签/自动事件类型 → 声纹类别
LABEL_TO_CAT = {"self_reload": "self_reload", "self_firemode": "self_firemode"}
TYPE_TO_CAT = {"footstep": "self_footstep", "gunshot": "self_gunshot",
               "action": "self_reload", "glass": "self_glass"}


def load_wav(path):
    """读 16bit PCM wav → [N, 2] float32 + 采样率。"""
    with wave.open(str(path), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise SystemExit(f"只支持 16bit PCM，得到 sampwidth={sw}")
    a = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch).astype(np.float32) / 32767.0
    return a, sr


def window_at(pcm, sr, t):
    """与运行时 vp_window 同几何：从整段音频里取 [t-0.05, t+0.13]。"""
    i0 = max(0, int((t - VP_PRE) * sr))
    i1 = min(len(pcm), int((t + VP_POST) * sr))
    if i1 - i0 < int(0.05 * sr):
        raise SystemExit(f"t={t}: 窗口太窄，检查时间点是否在音频范围内")
    win = pcm[i0:i1]
    if win.shape[1] == 1:
        win = np.stack([win[:, 0], win[:, 0]], axis=1)
    return win


def fp_at(pcm, sr, t):
    win = window_at(pcm, sr, t)
    if float(np.sqrt(np.mean(win ** 2))) < 1e-4:
        return None  # 窗口是静音（撤离后菜单期/断流），指纹是零向量垃圾
    return fingerprint(win, attack=VP_ATTACK)


def load_lib(path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_lib(path, lib):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(lib, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_add(args):
    pcm, sr = load_wav(args.file)
    fp = fp_at(pcm, sr, args.t)
    if fp is None:
        raise SystemExit(f"t={args.t}: 窗口是静音，没有可入库的声音")
    lib = load_lib(args.lib)
    lib.setdefault(args.cat, []).append(fp)
    save_lib(args.lib, lib)
    print(f"已入库: {args.cat} @ {args.t}s（库共 {sum(len(v) for v in lib.values())} 条）"
          f"  lib={args.lib}")


def cmd_build(args):
    sdir = Path(args.session)
    wav = sdir / "session.wav"
    if not wav.exists():
        raise SystemExit(f"找不到 {wav}")
    pcm, sr = load_wav(wav)

    # 1) 自标标签（r/b 按过键，时间最可信）
    candidates = []  # (t, cat, src)
    labels_f = sdir / "labels.jsonl"
    if labels_f.exists():
        for line in labels_f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cat = LABEL_TO_CAT.get(row["label"])
            if cat:
                candidates.append((row["t"], cat, f"label:{row['label']}"))

    # 2) 自动事件里规则已标 self 的（没被标签覆盖的才收，±0.5s 去重）
    auto_f = sdir / "auto_events.jsonl"
    n_auto = 0
    if auto_f.exists():
        for line in auto_f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("self"):
                continue
            cat = TYPE_TO_CAT.get(row["type"])
            if not cat:
                continue
            if any(abs(row["t"] - t) <= 0.5 for t, _, _ in candidates):
                continue  # 标签已覆盖
            candidates.append((row["t"], cat, f"auto:{row.get('self_reason', '') or 'rule'}"))
            n_auto += 1

    if not candidates:
        print("没有可用的自身声源：会话里既没有 r/b 自标标签，也没有规则标 self 的事件。")
        print("先开 collector 玩一局（按 R 换弹、走动），再回来 build。")
        return

    lib = load_lib(args.lib)
    added = {}
    skipped = 0
    for t, cat, src in sorted(candidates):
        fp = fp_at(pcm, sr, t)
        if fp is None:
            skipped += 1
            print(f"  ! 跳过 {cat} @ {t:.2f}s：窗口是静音（撤离后菜单期或断流）")
            continue
        lib.setdefault(cat, []).append(fp)
        added[cat] = added.get(cat, 0) + 1
        print(f"  + {cat:<16} @ {t:8.2f}s  ({src})")
    save_lib(args.lib, lib)
    print(f"\n入库 {sum(added.values())} 条（跳过静音 {skipped} 条，其中自动事件来源 "
          f"{n_auto} 条）: " + ", ".join(f"{k}={v}" for k, v in sorted(added.items())))
    print(f"库总量 {sum(len(v) for v in lib.values())} 条  lib={args.lib}")
    print("提醒：阈值还没标定，先用 match 子命令验证几条敌我样本再收紧 thresh。")


def cmd_match(args):
    pcm, sr = load_wav(args.file)
    fp = fp_at(pcm, sr, args.t)
    lib = load_lib(args.lib)
    if not lib:
        raise SystemExit(f"库是空的: {args.lib}，先 add/build")
    rows = []
    for cat, fps in lib.items():
        d = min(cosine_dist(fp, f) for f in fps)
        rows.append((d, cat))
    rows.sort()
    for d, cat in rows:
        print(f"  {cat:<20} d={d:.3f}")
    d0, cat0 = rows[0]
    verdict = "自己" if d0 <= args.thresh else "敌人/未知"
    print(f"\nt={args.t}s → 最近 {cat0} (d={d0:.3f})，阈值 {args.thresh} → 判定: {verdict}")


def cmd_list(args):
    lib = load_lib(args.lib)
    if not lib:
        print(f"库是空的: {args.lib}")
        return
    print(f"库: {args.lib}")
    for cat, fps in sorted(lib.items()):
        print(f"  {cat:<20} {len(fps)} 条")


def main():
    ap = argparse.ArgumentParser(description="自身声纹库工具（防线3）")
    ap.add_argument("--lib", default=DEFAULT_LIB, help="库文件路径")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="手动加一条指纹")
    p.add_argument("--file", required=True)
    p.add_argument("--t", type=float, required=True, help="事件时间点（秒）")
    p.add_argument("--cat", required=True, help="类别，如 self_footstep")
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("build", help="从 collector 会话建库")
    p.add_argument("--session", required=True, help="sessions/<时间戳> 目录")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("match", help="验证时间点是敌是我")
    p.add_argument("--file", required=True)
    p.add_argument("--t", type=float, required=True)
    p.add_argument("--thresh", type=float, default=0.30)
    p.set_defaults(fn=cmd_match)

    p = sub.add_parser("list", help="看库内容")
    p.set_defaults(fn=cmd_list)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

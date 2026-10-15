"""姿态重建：从 keys.jsonl 边沿流还原每个时刻的运动姿态。

五类：walk / run / jump / crouch / crawl
语义（三角洲用户键位实测 20261001-164045 定案）：
  - shift 点按切换冲刺（用户游戏内"切换疾跑"设置，按住直读只有 0.1s 边沿）
    冲刺中按 C/Z 会退出冲刺；蹲/趴时按 shift = 站起冲刺（清除蹲趴）
  - c 每次按下翻转蹲状态；蹲生效时清除趴和冲刺（游戏内蹲优先站立）
  - z 每次按下翻转趴状态；趴生效时清除蹲和冲刺
  - w/a/s/d 任一按住 → 移动中；全部松开后冲刺闩锁 1.0s 过期（停下=冲刺结束）
  - space 按下后 0.8s → jump
优先级：jump > crawl > crouch > run > walk > None（站立=无键）

产出 sessions/<id>/pose.jsonl：{"t", "pose"} 状态翻转序列。
2026-10-01 之前的老会话没监控 shift/z，跑/趴段标不出来，
pose 会给 unknown——属正常，重录的素材才有完整五类。
"""
import argparse
import json
from pathlib import Path

POSES = ["walk", "run", "jump", "crouch", "crawl"]
JUMP_HOLD = 0.8
RUN_GRACE = 1.0      # 全部移动键松开后冲刺闩锁保留时长
MOVE_KEYS = {"w", "a", "s", "d"}


def pose_timeline(keys_path):
    """keys.jsonl → [(t, pose)] 翻转序列（t 升序）。"""
    edges = [json.loads(l) for l in open(keys_path, encoding="utf-8")]
    edges.sort(key=lambda e: e["t"])
    state = {k: False for k in MOVE_KEYS | {"shift", "c", "z", "space"}}
    crouch = prone = run = False
    jump_until = -1.0
    run_until = -1.0     # 冲刺闩锁过期时刻
    last = None
    tl = []

    def pose_at(t):
        if t < jump_until:
            return "jump"
        if prone:
            return "crawl"
        if crouch:
            return "crouch"
        if run and (any(state[k] for k in MOVE_KEYS) or t < run_until):
            return "run"
        if any(state[k] for k in MOVE_KEYS):
            return "walk"
        return None

    for e in edges:
        k, down, t = e["key"], e["down"], e["t"]
        if k == "shift" and down:
            if crouch or prone:          # 蹲/趴中冲刺 = 站起冲刺
                crouch = prone = False
                run = True
            else:
                run = not run            # 切换冲刺
            run_until = t + RUN_GRACE
        elif k == "c" and down:
            crouch = not crouch
            if crouch:
                prone = False            # 蹲优先于趴（游戏内起立半蹲）
        elif k == "z" and down:
            prone = not prone
            if prone:
                crouch = False
                run = False
        elif k == "space" and down:
            jump_until = t + JUMP_HOLD
        elif k in state:
            was_moving = any(state[m] for m in MOVE_KEYS)
            state[k] = down
            if was_moving and not any(state[m] for m in MOVE_KEYS):
                run_until = t + RUN_GRACE   # 全松开：冲刺进入宽限期
        p = pose_at(t)
        if p != last:
            tl.append((t, p))
            last = p
    return tl


def pose_at(tl, t):
    """翻转序列 + 时刻 → 姿态（二分/线性扫，tl 很短直接扫）。"""
    p = None
    for tt, pp in tl:
        if tt <= t:
            p = pp
        else:
            break
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="+", help="sessions/ 下的会话目录名")
    ap.add_argument("--events", action="store_true",
                    help="顺便给 auto_events.jsonl 的脚步事件标姿态，输出 pose_events.jsonl")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent / "sessions"
    for sid in args.session:
        d = root / sid
        kp = d / "keys.jsonl"
        if not kp.exists():
            print(f"{sid}: 无 keys.jsonl，跳过")
            continue
        tl = pose_timeline(kp)
        out = d / "pose.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for t, p in tl:
                f.write(json.dumps({"t": round(t, 3), "pose": p},
                                   ensure_ascii=False) + "\n")
        from collections import Counter
        seg = Counter()
        ends = [t for t, _ in tl[1:]] + [max(tl[-1][0], edges_tmax(kp)) + 5.0]
        for (t0, p0), t1 in zip(tl, ends):
            if p0:
                seg[p0] += round(t1 - t0, 1)
        print(f"{sid}: {len(tl)} 次翻转 → {out.name}"
              f"（累计: {dict(seg)}）")
        if args.events:
            ep, ev_out = d / "auto_events.jsonl", d / "pose_events.jsonl"
            if not ep.exists():
                print(f"  无 auto_events.jsonl")
                continue
            n = 0
            with open(ev_out, "w", encoding="utf-8") as f:
                for l in open(ep, encoding="utf-8"):
                    e = json.loads(l)
                    if e.get("type") != "footstep":
                        continue
                    f.write(json.dumps(
                        {"t": e["t"], "pose": pose_at(tl, e["t"]),
                         "self": e.get("self", False)},
                        ensure_ascii=False) + "\n")
                    n += 1
            print(f"  {n} 条脚步事件 → {ev_out.name}")


def t0_last(tl):
    return tl[-1][0]


def edges_tmax(keys_path):
    m = 0.0
    for l in open(keys_path, encoding="utf-8"):
        t = json.loads(l)["t"]
        if t > m:
            m = t
    return m


if __name__ == "__main__":
    main()

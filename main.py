"""Delta Audio Radar v0.1 —— 听声辨位系统骨架

用法：
  python main.py --demo          离线合成信号验证 DSP 链路（无需任何音频设备）
  python main.py --loopback      抓取系统输出（游戏/录像声音），开调试面板
  python main.py --file x.wav    用 wav 文件回放验证（可配合三角洲录像提取的音频）
  python main.py --list          列出可用 loopback 设备
"""
import argparse
import json
import sys
import threading
import queue

ROOT = __file__
sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")


def load_cfg():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def run_demo(cfg):
    """离线验证：合成信号 -> 引擎 -> 打印事件，并核对预期。"""
    import tempfile
    from pathlib import Path

    from capture.synth import demo_blocks
    from dsp.pipeline import Engine
    from dsp.guntype import fingerprint, cosine_dist
    from dsp.selfsound import vp_window

    # 防线3验收前置：先用同一合成信号攒一条自身声纹（10.4s 自己的脚步），
    # 顺便量出最难分的敌步（2.0s 正中脚步）到它的距离，阈值取一半——
    # 同一几何下重放距离可复现，这个阈值在干净样本上必然分开敌我。
    eng0 = Engine(cfg)
    win_self = win_enemy = None
    for t, block in demo_blocks(cfg["block_size"]):
        eng0.process(t, block)
        if win_enemy is None and t >= 2.2:
            win_enemy = vp_window(eng0.buf, 2.0)
        if t >= 10.6:
            win_self = vp_window(eng0.buf, 10.4)
            break
    fp_self = fingerprint(win_self, attack=0.08)
    d_enemy = cosine_dist(fingerprint(win_enemy, attack=0.08), fp_self)
    vp_thresh = round(d_enemy / 2, 3)
    tmp_lib = Path(tempfile.gettempdir()) / "delta_demo_selfvoice.json"
    tmp_lib.write_text(json.dumps({"self_footstep": [fp_self]}), encoding="utf-8")
    cfg = dict(cfg, self_voiceprint={"enabled": True, "lib": str(tmp_lib),
                                     "thresh": vp_thresh})

    engine = Engine(cfg)
    events = []
    for t, block in demo_blocks(cfg["block_size"]):
        # 防线1扩展验收注入：8.0s 前按 W（移动键掩蔽），6.0s 前点左键（开火掩蔽）
        if 7.8 <= t < 8.0:
            engine.on_hotkey("w")
        if 5.9 <= t < 6.0:
            engine.on_mouse("left")
        snap = engine.process(t, block)
        if snap["event"]:
            ev = snap["event"]
            events.append((t, ev["type"], round(ev["pan"], 2),
                           snap["tier"], snap["masked"], ev.get("self", False),
                           ev.get("self_reason", "")))
            print(f"[{t:5.2f}s] {ev['type']:<10} pan={ev['pan']:+.2f} "
                  f"档={snap['tier']} masked={snap['masked']} self={ev.get('self', False)}"
                  f" {ev.get('self_reason', '')}")

    print("\n--- 核对预期 ---")
    ok = True
    expect = [
        (1.0, "footstep"), (2.0, "footstep"), (3.2, "footstep"),
        (5.0, "footstep"),   # 嘈杂中仍能触发
        (6.0, "gunshot"),
        (9.2, "footstep"),   # 6米级静步，静步档捕获
        (9.8, "glass"),      # 碎玻璃
    ]
    for et, etype in expect:
        hit = any(abs(t - et) < 0.3 and ty == etype for t, ty, *_ in events)
        print(f"  {et:4.1f}s {etype:<10} {'OK' if hit else 'MISS'}")
        ok &= hit
    suppressed = not any(abs(t - 6.3) < 0.3 and ty == "footstep" for t, ty, *_ in events)
    print(f"  6.3s 掩蔽窗内脚步被丢弃 {'OK' if suppressed else 'FAIL'}")
    ok &= suppressed
    # 自身脚步（10.4s，最响+居中+单声道）应标记 self，敌步不应误标
    # （热键/鼠标掩蔽是故意标自，不算误标，从敌步检查里排除）
    own = [e for e in events if abs(e[0] - 10.4) < 0.3 and e[1] == "footstep"]
    own_self = bool(own) and own[0][5] is True
    enemy_self = any(e[5] is True and not str(e[6]).startswith("hotkey")
                     for e in events if e[1] == "footstep" and abs(e[0] - 10.4) >= 0.3)
    print(f"  10.4s 自身脚步标记 self {'OK' if own_self else 'FAIL'}"
          f" | 敌步未误标 {'OK' if not enemy_self else 'FAIL'}")
    ok &= own_self and not enemy_self
    # 防线1扩展：移动键掩蔽（8.0s 脚步应标 hotkey:move）+ 鼠标开火掩蔽（6.0s 枪声标 hotkey:fire）
    move = [e for e in events if abs(e[0] - 8.1) < 0.35 and e[1] == "footstep"]
    fire = [e for e in events if abs(e[0] - 6.1) < 0.35 and e[1] == "gunshot"]
    move_ok = bool(move) and move[0][5] is True and str(move[0][6]) == "hotkey:move"
    fire_ok = bool(fire) and fire[0][5] is True and str(fire[0][6]) == "hotkey:fire"
    print(f"  8.0s 移动键掩蔽 {'OK' if move_ok else 'FAIL'}"
          f" | 6.0s 鼠标开火掩蔽 {'OK' if fire_ok else 'FAIL'}")
    ok &= move_ok and fire_ok
    # 防线3声纹库：10.4s 必须命中 voiceprint 理由；敌步不得被声纹误标
    vp_own = any(str(e[6]).startswith("voiceprint")
                 for e in events if abs(e[0] - 10.4) < 0.3)
    vp_enemy = any(str(e[6]).startswith("voiceprint")
                   for e in events if e[1] == "footstep" and abs(e[0] - 10.4) >= 0.3)
    print(f"  10.4s 声纹库命中 {'OK' if vp_own else 'FAIL'}"
          f" | 敌步未被声纹误标 {'OK' if not vp_enemy else 'FAIL'}"
          f"  [d_enemy={d_enemy:.3f}, thresh={vp_thresh}]")
    ok &= vp_own and not vp_enemy
    tmp_lib.unlink(missing_ok=True)
    print("\nDEMO", "PASS" if ok else "FAIL")
    return ok


def run_source(cfg, source, ui=True, seconds=None, show_self_override=False):
    from dsp.pipeline import Engine
    engine = Engine(cfg)

    hm = cfg.get("hotkey_mask") or {}
    if hm.get("enabled", True) and (hm.get("keys") or hm.get("mouse")):
        try:
            import keyboard
            desc = []
            for key, spec in hm["keys"].items():
                keyboard.on_press_key(key, lambda _e, k=key: engine.on_hotkey(k))
                desc.append(f"{key.upper()}={spec.get('label', '?')}")
            print("热键掩蔽已挂载: " + ", ".join(desc))
        except ImportError:
            print("!! 未安装 keyboard 库，键盘掩蔽不可用（pip install keyboard）")
        try:
            import mouse
            btns = {b.lower() for b, s in (hm.get("mouse") or {}).items()
                    if isinstance(s, dict) and not b.startswith("_")}

            def _mcb(ev, _btns=btns):
                # on_click 按下和抬起各触发一次，会把松手后的 0.3s 也吞掉；
                # 只认 down 边沿。未配置的键 on_mouse 内部也会拒。
                # mouse.hook 会把移动/滚轮事件也喂进来（MoveEvent 等，不是 dict）
                if not hasattr(ev, "get"):
                    return
                if ev.get("event_type") == "down" and ev.get("button", "").lower() in _btns:
                    engine.on_mouse(ev["button"])

            if btns:
                mouse.hook(_mcb)
                print("鼠标掩蔽已挂载: " + ", ".join(sorted(btns)))
        except ImportError:
            print("!! 未安装 mouse 库，鼠标掩蔽不可用（pip install mouse）")
    q = queue.Queue(maxsize=50)

    def worker():
        while True:
            item = source.read()
            if item is None:
                q.put(None)
                break
            t, frame = item
            try:
                q.put(engine.process(t, frame), timeout=0.1)
            except queue.Full:
                pass
        source.close()

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    def _shutdown():
        # 收尾：等 worker 走完 EOF，卸键盘/鼠标钩子——不然解释器销毁时
        # daemon 线程写着 stderr 被腰斩，Windows 上刷一屏退出噪音
        th.join(timeout=2)
        try:
            import keyboard
            keyboard.unhook_all()
        except Exception:
            pass
        try:
            import mouse
            mouse.unhook_all()
        except Exception:
            pass

    if not ui:
        # 输出层过滤：config output.show_self 或 --show-self 才显示自己的声音
        show_self = cfg.get("output", {}).get("show_self", False) or show_self_override
        try:
            while True:
                snap = q.get()
                if snap is None:
                    break
                ev = snap.get("event")
                if ev and ev["type"] in ("footstep", "gunshot"):
                    if ev.get("self") and not show_self:
                        continue  # 输出层过滤：自己的声音不显示（事件已带标记落盘）
                    pan = ev["pan"]
                    side = "左" if pan < -0.35 else ("右" if pan > 0.35 else "中")
                    reason = ev.get("self_reason", "")
                    self_tag = " [自己]" + (f"({reason})" if reason else "") if ev.get("self") else ""
                    print(f"[{snap['t']:7.2f}s] {ev['type']:<9} "
                          f"{side} pan={pan:+.2f} 档={snap['tier']} "
                          f"平坦={snap['feats']['low_flat']:.3f}{self_tag}", flush=True)
                if seconds and snap.get("t", 0) >= seconds:
                    break
        except KeyboardInterrupt:
            pass
        _shutdown()
        return

    from ui.panel import Panel
    panel = Panel(q, device_name=getattr(source, "device_name", "file"),
                  show_self=cfg.get("output", {}).get("show_self", False)
                  or show_self_override)
    panel.run()
    _shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--loopback", action="store_true")
    ap.add_argument("--file")
    ap.add_argument("--device", default=None, help="loopback 设备名关键词")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--headless", action="store_true", help="不开面板，事件直接打印到控制台")
    ap.add_argument("--seconds", type=float, default=None, help="只处理前 N 秒（调试用）")
    ap.add_argument("--skip", type=float, default=0.0, help="跳过文件前 N 秒")
    ap.add_argument("--show-self", action="store_true",
                    help="显示自己发出的声音（默认按 config output.show_self 过滤掉）")
    args = ap.parse_args()

    if args.list:
        from capture.source import list_loopback_devices
        for name in list_loopback_devices():
            print(name)
        return

    cfg = load_cfg()

    if args.demo:
        sys.exit(0 if run_demo(cfg) else 1)

    if args.file:
        from capture.source import FileSource
        src = FileSource(args.file, sr=cfg["sample_rate"], block=cfg["block_size"],
                         skip=args.skip)
    elif args.loopback:
        from capture.source import LoopbackSource
        src = LoopbackSource(device_keyword=args.device,
                             sr=cfg["sample_rate"], block=cfg["block_size"])
        print(f"loopback 设备: {src.device_name}")
    else:
        ap.print_help()
        return

    try:
        run_source(cfg, src, ui=(not args.headless), seconds=args.seconds,
                   show_self_override=args.show_self)
    except KeyboardInterrupt:
        print("退出")


if __name__ == "__main__":
    main()

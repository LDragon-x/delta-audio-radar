"""实时音频过道：loopback 抓游戏声 → 多频段 FX（缩枪声/抬脚步）→ 播放到耳机。

部署前提（重要）：输出设备不能和 loopback 抓取的设备是同一个——
否则我们的输出会被自己的输入抓回去形成回声。正确接法：
  1. 装 VB-Cable（虚拟声卡），游戏音频输出设为 "CABLE Input"
  2. 本工具：--in-device CABLE（loopback 抓 CABLE Output）
  3. 本工具：--out-device 你的耳机
  声音路径：游戏 → CABLE → 本程序(处理) → 耳机，程序对游戏零注入。

用法：
  python tool/monitor.py --list                     # 列出所有音频设备
  python tool/monitor.py --in-device "CABLE" --out-device "GAMING HEADSET"
  python tool/monitor.py --fx-off                   # 旁路对比模式（原声直通）
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsp.livefx import MultibandFX, BLOCK

SR = 48000


def list_devices():
    import soundcard as sc
    print("== 输出设备（可作为 --out-device）==")
    for s in sc.all_speakers():
        print(" ", s.name)
    print("== Loopback 输入（可作为 --in-device）==")
    for m in sc.all_microphones(include_loopback=True):
        print(" ", m.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-device", default=None)
    ap.add_argument("--out-device", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fx-off", action="store_true", help="旁路：不做处理直接转发")
    ap.add_argument("--low-boost-db", type=float, default=5.0)
    args = ap.parse_args()

    if args.list:
        list_devices()
        return

    import soundcard as sc
    import numpy as np

    src_name = None
    for m in sc.all_microphones(include_loopback=True):
        if args.in_device and args.in_device in m.name:
            src_name = m.name
            break
    if src_name is None and args.in_device is None:
        for m in sc.all_microphones(include_loopback=True):
            if "loopback" in m.name.lower():
                src_name = m.name
                break
    dst_name = None
    for s in sc.all_speakers():
        if args.out_device and args.out_device in s.name:
            dst_name = s.name
            break
    if dst_name is None:
        dst_name = str(sc.default_speaker().name)
    if src_name and args.in_device and args.in_device.lower() in dst_name.lower():
        print("!! 警告：输入与输出疑似同一设备，会产生回声反馈。请用 VB-Cable 路由。")
        return

    rec = sc.get_microphone(src_name, include_loopback=True).recorder(
        samplerate=SR, channels=2, blocksize=BLOCK)
    player = sc.get_speaker(dst_name).player(samplerate=SR, channels=2, blocksize=BLOCK)
    rec.__enter__()
    player.__enter__()
    fx = MultibandFX(low_boost_db=args.low_boost_db)
    mode = "旁路" if args.fx_off else "FX(缩枪声/抬脚步)"
    print(f"音频过道已启动 [{mode}]\n  输入: {src_name}\n  输出: {dst_name}\n"
          f"  理论延迟 ≈ {BLOCK/SR*1000:.0f}ms + 声卡缓冲\nCtrl+C 停止")
    try:
        while True:
            frames = rec.record(numframes=BLOCK)
            if args.fx_off:
                out = np.asarray(frames, dtype=np.float32)
            else:
                out = fx.process(np.asarray(frames, dtype=np.float32))
                if len(out) == 0:
                    continue
            player.play(out)
    except KeyboardInterrupt:
        pass
    finally:
        rec.__exit__(None, None, None)
        player.__exit__(None, None, None)
        print("已停止")


if __name__ == "__main__":
    main()

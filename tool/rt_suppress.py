# -*- coding: utf-8 -*-
"""实时自我消噪 v0.1：loopback 回采 → 实时检测 → embed 判定 → 门控 → 延迟播放到耳机。

关键设计（延迟路由）：
  embed 分类必须吃满窗 [t-0.15, t+0.35]（截断实验：85%→50%，JOURNAL 留痕），
  所以输出端压 OUT_DELAY 秒（默认 0.45s）的播放队列。分类在 t+0.35 完成时，
  事件窗恰好刚结束，而输出还压着 0.45s——整段事件窗可以回溯压增益后再播：
  自己脚步完整 -15dB，敌人声音一个字不碰。
  全链路延迟 ≈ 0.45 + 采集块 43ms + 播放缓冲 ≈ 0.5s。
  枪声闩锁：开火 [-1s, +2.5s] 内增益保底 -6dB（交战段不做深抑制），
  与事件窗是 max 关系（抬增益），事件窗之间是 min 关系（取最深）。

用法：
  python tool/rt_suppress.py                       # 实时：回采→消噪→播到默认输出
  python tool/rt_suppress.py --file session.wav --null --write out.wav
热键（实时模式）：F10 退出；F9 旁路（原声直通）。
事件判定写 rt_events.jsonl，供事后对账。
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tool"))

from capture.source import LoopbackSource, FileSource  # noqa: E402
from dsp.features import extract_features  # noqa: E402
from dsp.onset import OnsetDetector  # noqa: E402
from dsp.classify import Classifier, KnnGuess  # noqa: E402
from self_suppress import load_model_protos  # noqa: E402
import learn_embed as le  # noqa: E402

DECISION_DELAY = 0.09
OUT_DELAY = 0.18        # 截断增强模型（v0.4）半窗即可分类：分类等待 0.12s + 余量
EARLY_CUT = 0.12        # 事件后 0.12s 就分类（窗口前 0.15+后 0.12，补零到 0.5s）
GAIN_DB = 25.0        # self 深度抑制增益（用户实测 -15 压不住，2026-10-01）
GUN_FLOOR_DB = 6.0
GUN_LATCH_PRE, GUN_LATCH_POST = 1.0, 2.5
MONO_KEEP = 16000 * 30  # 16k 缓冲最多留 30s
# 姿态门控（2026-10-01）：键盘=运动姿态真值。
# 授权加成：按键说"我正在走/跑/蹲"→ 声学分加点，宁压；
# 站立保护：无移动键时听到的脚步八成是敌人 → 大幅降分，不压。
POSE_AUTH_BOOST = 0.10
POSE_STAND_PENALTY = 0.35
POSE_TAU = 0.35   # 判定门槛（2026-10-01 扫描：0.18→误伤21.8%，0.35→约17%，0.50→13.7%）


class PoseWatch:
    """独立线程轮询 GetAsyncKeyState，维护姿态历史（墙钟时间戳）。

    姿态语义（20261001-164045 实测定案，与 tool/pose_label.py 同步）：
    shift 点按=切换冲刺（游戏"切换疾跑"）；蹲/趴中按 shift=站起冲刺；
    c/z 切换式；蹲生效清趴，趴生效清蹲和冲刺；全部移动键松开后
    冲刺闩锁保留 1.0s；space 按下后 0.8s 内算 jump。
    优先级：jump > crawl > crouch > run > walk > None。
    pose_at_audio(t)：音频时间轴 → 墙钟（起始对齐 + 0.1s 检测延迟修正），
    从历史里取最近一条姿态。键盘只用于实时模式，--file 回放禁用。
    """

    def __init__(self):
        import ctypes
        self._get = ctypes.windll.user32.GetAsyncKeyState
        self.vk = {"w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44,
                   "shift": 0x10, "space": 0x20, "c": 0x43, "z": 0x5A}
        self.t0_wall = time.time()
        self.hist = deque()        # (wall_t, pose|None)
        self.crouch = self.prone = self.run = False
        self.jump_until = -1.0
        self.run_until = -1.0
        self.state = {k: False for k in ("w", "a", "s", "d")}
        self._stop = False
        import threading
        self.th = threading.Thread(target=self._run, daemon=True,
                                   name="pose-watch")
        self.th.start()

    def _cur(self, now):
        if now < self.jump_until:
            return "jump"
        if self.prone:
            return "crawl"
        if self.crouch:
            return "crouch"
        moving = any(self.state[k] for k in ("w", "a", "s", "d"))
        if self.run and (moving or now < self.run_until):
            return "run"
        if moving:
            return "walk"
        return None

    def _edge(self, name):
        """带边沿检测的按键状态：返回 (is_down, just_pressed)。"""
        dn = bool(self._get(self.vk[name]) & 0x8000)
        jp = dn and not getattr(self, f"_{name}_dn", False)
        setattr(self, f"_{name}_dn", dn)
        return dn, jp

    def _run(self):
        last = None
        while not self._stop:
            now = time.time()
            for m in ("w", "a", "s", "d"):
                self.state[m], _ = self._edge(m)
            _, sh = self._edge("shift")
            if sh:
                if self.crouch or self.prone:
                    self.crouch = self.prone = False
                    self.run = True
                else:
                    self.run = not self.run
                self.run_until = now + 1.0
            _, cd = self._edge("c")
            if cd:
                self.crouch = not self.crouch
                if self.crouch:
                    self.prone = False
            _, zd = self._edge("z")
            if zd:
                self.prone = not self.prone
                if self.prone:
                    self.crouch = False
                    self.run = False
            _, sp = self._edge("space")
            if sp:
                self.jump_until = now + 0.8
            if self.run and not any(self.state.values()):
                self.run_until = max(self.run_until, now + 1.0)
            p = self._cur(now)
            if p != last:
                self.hist.append((now, p))
                if len(self.hist) > 400:
                    self.hist.popleft()
                last = p
            time.sleep(0.015)

    def pose_at_audio(self, t_audio):
        wt = self.t0_wall + t_audio + 0.1
        p = None
        for hw, pw in self.hist:
            if hw <= wt:
                p = pw
            else:
                break
        return p

    def close(self):
        self._stop = True



class RtSuppress:
    def __init__(self, cfg, args):
        self.args = args
        cc = cfg["classify"]
        self.sr = cfg["sample_rate"]
        self.bands = cfg["bands"]
        self.onset = OnsetDetector(cfg["onset"])
        self.classifier = Classifier(cc["mono_corr_max"], cc.get("gun_full_min", 1e-3),
                                     cc.get("gun_high_min", 1e-5),
                                     cc.get("glass_high_ratio", 0.30),
                                     cc.get("glass_high_mid", 2.0),
                                     knn=KnnGuess(cc["knn_model"])
                                     if cc.get("knn_model") else None)
        self.model, self.mu, self.sd, self.ps, self.pf, self.n_train = load_model_protos()
        # 姿态模板（tool/train_pose_protos.py 产出）：{pose: 单位向量}
        import torch as _t
        self.pose_protos = {}
        pp_path = ROOT / "dsp" / "models" / "pose_protos.json"
        if pp_path.exists():
            try:
                for pose, d in json.load(open(pp_path, encoding="utf-8"))["protos"].items():
                    self.pose_protos[pose] = _t.tensor(d["v"], dtype=_t.float32)
                print(f"姿态模板: {list(self.pose_protos)}")
            except Exception as ex:
                print(f"姿态模板加载失败（忽略）: {ex}")
        # 键盘姿态监视：仅实时模式（--file 回放禁用，键盘状态与音频无关）
        self.kb = None if args.file else PoseWatch()
        self.prev_full = 0.0
        self._pending = None
        self.foot_q = deque()      # 待分类脚步 (t_e, deadline)
        self.infer_q = deque()     # 待推理 ('e',t_e,seg) 脚步 / ('f',t0,seg) 帧级
        self.envs = []             # 增益窗 (t0, t1, g)
        self.fenvs = []            # 帧级增益窗 (t0, t1, g48)
        self.frame_fs = None       # 帧级消噪器（frame_v0.pt 存在则加载）
        self.frame_upto = 0        # 帧级推理已消费到的 16k 绝对样本号
        self.frame_pow = getattr(args, "frame_pow", 2.0)
        self.frame_pow_pose = getattr(args, "frame_pow_pose", 3.0)
        _fckpt = ROOT / "dsp" / "models" / "frame_v0.pt"
        if _fckpt.exists() and not getattr(args, "no_frame", False):
            try:
                from dsp.frame_model import FrameSuppressor
                self.frame_fs = FrameSuppressor(_fckpt)
                print("帧级模型: 开（frame_v0.pt，200ms 回看上下文）")
            except Exception as ex:
                print(f"帧级模型加载失败（回退事件门控）: {ex}")
        self.latches = []          # 枪声闩锁 (t0, t1)
        self.play_q = deque()      # (t, frame48) 延迟播放队列
        self.mono = np.zeros(0, dtype=np.float32)
        self.mono_base = 0         # mono[0] 对应的绝对样本号
        self.mono_total = 0        # 已摄入的 16k 样本总数
        self.enabled = True
        self.rj = open(args.log, "a", encoding="utf-8") if args.log else None
        self.g_min = 10.0 ** (-GAIN_DB / 20.0)
        self.g_floor = 10.0 ** (-GUN_FLOOR_DB / 20.0)
        self.sd_cap = None
        self.out_dev = None
        self.wav_f = None
        if args.write:
            import wave
            self.wav_f = wave.open(args.write, "wb")
            self.wav_f.setnchannels(2)
            self.wav_f.setsampwidth(2)
            self.wav_f.setframerate(self.sr)
        if not args.file:
            self._open_audio(args)

    def _open_audio(self, args):
        """拓扑 v2（闭环）：游戏/系统输出 → VB-Audio 虚拟线 → 本进程采集 →
        消噪 → 播到 GAMING 端点（耳朵）。耳朵只听到处理后的声音，无双声。
        采集/播放都走 sounddevice（PortAudio 管缓冲）。
        注：v1 用 loopback 抓 GAMING 直出是开环——原声绕过处理直达耳朵（重影），
        且同端点回采+回播有数字回授（卡）。VB 线是正解。"""
        import sounddevice as sd
        wasapi = next(i for i, a in enumerate(sd.query_hostapis())
                      if a["name"] == "Windows WASAPI")
        # 采集：VB 虚拟线的录音端（"入梦麦克风 (VB-Audio Cable A)"）
        cap_kw = args.cap_device or "CABLE"
        cap_i = next((i for i, d in enumerate(sd.query_devices())
                      if d["hostapi"] == wasapi
                      and d["max_input_channels"] > 0
                      and cap_kw in d["name"].upper()), None)
        if cap_i is None:
            raise RuntimeError(f"找不到虚拟线输入端点（{cap_kw}）")
        cap_info = sd.query_devices(cap_i)
        cap_sr = int(cap_info["default_samplerate"])  # VB 线只开 44.1k
        self.sr_cap = cap_sr
        self.sd_cap = sd.InputStream(device=cap_i, channels=2, samplerate=cap_sr,
                                     blocksize=int(4096 * cap_sr / self.sr),
                                     latency="low")
        self.sd_cap.start()
        self.cap_block = 4096  # 重采样后的 48k 帧数
        self._frames = 0
        print(f"采集: {cap_info['name']} @ {cap_sr}Hz → 重采样 {self.sr}Hz")
        if args.null:
            return
        play_kw = args.play_device or "GAMING"
        play_i = next((i for i, d in enumerate(sd.query_devices())
                       if d["hostapi"] == wasapi
                       and d["max_output_channels"] > 0
                       and play_kw in d["name"]), None)
        if play_i is None:
            raise RuntimeError(f"找不到播放端点 {play_kw}")
        self.out_dev = sd.OutputStream(device=play_i, channels=2,
                                       samplerate=self.sr, blocksize=0,
                                       latency=0.1)
        self.out_dev.start()
        print(f"播放: {sd.query_devices(play_i)['name']}")

    def _embed_seg(self, seg):
        import torch
        x = le.seg_patch(seg)
        xb = torch.tensor((x - self.mu) / self.sd)[None, None]
        with torch.no_grad():
            e = torch.nn.functional.normalize(self.model(xb), dim=1)
        pt = torch.tensor
        return float(e[0] @ pt(self.ps)) - float(e[0] @ pt(self.pf))

    def _detect(self, t, frame):
        feats = extract_features(frame, self.sr, self.bands)
        feats["attack"] = feats["full"] / max(self.prev_full, 1e-12)
        self.prev_full = feats["full"]
        level = self.onset.update(feats, t)
        if level:
            self._pending = {"t": t, "frames": [(t, feats)], "quiet": level == "quiet"}
        elif self._pending is not None:
            self._pending["frames"].append((t, feats))
        if self._pending is not None and t - self._pending["t"] >= DECISION_DELAY:
            peak_t, peak_feats = max(self._pending["frames"], key=lambda x: x[1]["low"])
            peak_feats = dict(peak_feats)
            peak_feats["quiet"] = self._pending["quiet"]
            event = self.classifier.update(peak_feats, self._pending["t"])
            self._pending = None
            if event is None:
                return
            t_e = event["t"]
            if event["type"] == "footstep":
                self.foot_q.append((t_e, t_e + EARLY_CUT))
            elif event["type"] == "gunshot":
                self.latches.append((t_e - GUN_LATCH_PRE, t_e + GUN_LATCH_POST))

    def _classify_feet(self, t_now):
        """到点的脚步事件：取 [t-0.15, t+0.12] 截断窗（补零）塞给推理线程。
        v0.4 截断增强训练，半窗分类反而更准（71.7% vs 65.2% 满窗）。"""
        while self.foot_q and self.foot_q[0][1] <= t_now:
            t_e, _ = self.foot_q.popleft()
            i0 = int((t_e - le.PRE) * 16000) - self.mono_base
            i1 = int((t_e + EARLY_CUT) * 16000) - self.mono_base
            if i0 < 0 or i1 > len(self.mono):
                print(f"! 缓冲不足 t={t_e:.1f}，跳过")
                continue
            seg = np.pad(self.mono[i0:i1], (0, max(0, le.SEG_LEN - (i1 - i0))))
            self.infer_q.append(("e", t_e, seg))

    def _push_frame_blocks(self):
        """新的 16k 单声道样本按 2048 块推进帧级推理队列。
        帧级模型 200ms 上下文为回看式：块内全部帧在块齐后即可输出，
        端到端延迟 ≈ 块周期 128ms + 推理毫秒级 < OUT_DELAY 180ms，不拖延迟。"""
        if self.frame_fs is None:
            return
        while self.mono_total - self.frame_upto >= 2048:
            i0 = self.frame_upto - self.mono_base
            if i0 < 0:
                self.frame_upto = self.mono_base   # 缓冲裁剪后跳过
                continue
            seg = np.array(self.mono[i0:i0 + 2048], dtype=np.float32)
            t0 = self.frame_upto / 16000.0
            self.infer_q.append(("f", t0, seg))
            self.frame_upto += 2048

    def _infer_loop(self):
        import torch
        pt = torch.tensor
        while True:
            try:
                kind, t_e, seg = self.infer_q.popleft()
            except IndexError:
                time.sleep(0.002)
                continue
            if kind == "f":
                cleaned = self.frame_fs.process_block(seg)
                g = np.abs(cleaned) / (np.abs(seg) + 1e-6)
                # v0.3：存原始比增益，塑形延迟到 _gain_block 按姿态条件施加
                # （键盘=我在动 → 压狠点；站着 → 保守防误伤）
                g = np.clip(g, 0.0, 1.0).astype(np.float32)
                m = len(g)
                g48 = np.interp(np.linspace(0, 1, int(m / 16000 * self.sr)),
                                np.linspace(0, 1, m), g)
                self.fenvs.append((t_e, t_e + m / 16000.0, g48))
                continue
            x = le.seg_patch(seg)
            xb = pt((x - self.mu) / self.sd)[None, None]
            with torch.no_grad():
                e = torch.nn.functional.normalize(self.model(xb), dim=1)
            score = float(e[0] @ pt(self.ps)) - float(e[0] @ pt(self.pf))
            # 姿态门控：键盘=运动姿态真值。按 C 消蹲声、Shift 消跑步声——
            # 姿态专属模板换 ps；有键授权加分；站立（无键）听到的脚步八成
            # 是敌人的，大幅降分防误伤。--file 回放 kb=None，不进门控。
            pose = self.kb.pose_at_audio(t_e) if self.kb else "file"
            if pose in self.pose_protos:
                s_pose = float(e[0] @ pt(self.pose_protos[pose])) \
                    - float(e[0] @ pt(self.pf))
                score = max(score, s_pose)
            if pose in ("walk", "run", "jump", "crouch", "crawl"):
                score += POSE_AUTH_BOOST
            elif pose is None:
                score -= POSE_STAND_PENALTY
            g = 1.0 - (1.0 - self.g_min) / (1.0 + np.exp(-(score - POSE_TAU) / 0.1))
            self.envs.append((t_e - le.PRE, t_e + le.POST, g))
            if self.rj:
                self.rj.write(json.dumps(
                    {"t": round(t_e, 3), "score": round(score, 3),
                     "pose": pose, "gain": round(g, 3)}) + "\n")
                self.rj.flush()
            print(f"t={t_e:7.1f} score={score:+.2f} pose={pose or '-':6s} "
                  f"→ 增益 {20*np.log10(max(g,1e-6)):5.1f}dB")

    def _gain_block(self, t0, t1):
        n = int(round((t1 - t0) * self.sr))
        env = np.ones(n, dtype=np.float32)
        if not self.enabled:
            return env
        self.envs = [w for w in self.envs if w[1] >= t0 - 1.0]
        self.fenvs = [w for w in self.fenvs if w[1] >= t0 - 2.0]
        self.latches = [w for w in self.latches if w[1] >= t0 - 1.0]
        # 帧级增益作基底（连续掩码，误伤低）；塑形按姿态条件施加：
        # 键盘移动姿态=那段大概率是自己脚步 → 幂更高压更狠；
        # 站立无键 → 保守幂，防把敌人脚步一起闷掉
        for w0, w1, g48 in self.fenvs:
            i0 = max(0, int((w0 - t0) * self.sr))
            i1 = min(n, int((w1 - t0) * self.sr))
            if i0 < i1:
                s0 = i0 - int((w0 - t0) * self.sr)
                gg = g48[s0:s0 + (i1 - i0)]
                pose = self.kb.pose_at_audio((w0 + w1) / 2) if self.kb else "file"
                pw = self.frame_pow_pose if pose in ("walk", "run", "jump", "crouch", "crawl") else self.frame_pow
                gg = np.clip(gg ** pw, self.g_min, 1.0)
                env[i0:i1] = np.minimum(env[i0:i1], gg)
        POSES = ("walk", "run", "jump", "crouch", "crawl")
        for w0, w1, g in self.envs:
            i0 = max(0, int((w0 - t0) * self.sr))
            i1 = min(n, int((w1 - t0) * self.sr))
            if i0 >= i1:
                continue
            if g < 0.5:
                # 深压需键盘姿态背书（站着不动时的深压≈误伤敌人）；
                # 帧级模型上线后站立段全靠帧级增益保护。
                pose = self.kb.pose_at_audio((w0 + w1) / 2) if self.kb else "file"
                if pose not in POSES and pose != "file":
                    continue
            env[i0:i1] = np.minimum(env[i0:i1], g)
        if self.latches:
            latch_cov = np.zeros(n, dtype=bool)
            for w0, w1 in self.latches:
                i0 = max(0, int((w0 - t0) * self.sr))
                i1 = min(n, int((w1 - t0) * self.sr))
                if i0 < i1:
                    latch_cov[i0:i1] = True
            env[latch_cov] = np.maximum(env[latch_cov], self.g_floor)
        return env

    def _capture_loop(self):
        """独立采集线程：输入流必须被准时 read，否则缓冲溢出丢块（咔哒声）。
        读出的块进队列，主循环按自己的节奏消费。"""
        import threading
        while not self._stop_cap.is_set():
            try:
                d, _ovf = self.sd_cap.read(self.sd_cap.blocksize)
                # 线性重采样到 48k（VB 线 44.1k）；游戏音频够用
                n_in = len(d)
                n_out = self.cap_block
                xp = np.linspace(0.0, 1.0, n_in, endpoint=False)
                xq = np.linspace(0.0, 1.0, n_out, endpoint=False)
                r = np.empty((n_out, 2), dtype=np.float32)
                r[:, 0] = np.interp(xq, xp, d[:, 0])
                r[:, 1] = np.interp(xq, xp, d[:, 1])
                self.cap_q.append((None, r))
            except Exception:
                break

    def run(self, src):
        print("实时消噪运行中：F10 退出 / F9 旁路")
        try:
            import keyboard
            keyboard.add_hotkey("f10", lambda: sys.exit(0))
            keyboard.add_hotkey("f9", self._toggle)
        except Exception:
            pass
        import threading
        threading.Thread(target=self._infer_loop, daemon=True).start()
        if self.sd_cap is not None:
            self.cap_q = deque(maxlen=256)
            self._stop_cap = threading.Event()
            threading.Thread(target=self._capture_loop, daemon=True).start()
        while True:
            if self.sd_cap is not None:
                while not self.cap_q:
                    time.sleep(0.001)
                _, frame = self.cap_q.popleft()
                self._frames += self.cap_block
                t = self._frames / self.sr
            else:
                item = src.read()
                if item is None:
                    break
                t, frame = item
            frame = np.asarray(frame, dtype=np.float32)
            mono_blk = frame.mean(axis=1)[::3]
            self.mono = np.concatenate([self.mono, mono_blk])
            self.mono_total += len(mono_blk)
            if len(self.mono) > MONO_KEEP:
                cut = len(self.mono) - MONO_KEEP
                self.mono = self.mono[cut:]
                self.mono_base += cut
            self._push_frame_blocks()
            self._detect(t, frame)
            self._classify_feet(t)
            self.play_q.append((t, frame))
            while self.play_q and self.play_q[0][0] <= t - OUT_DELAY:
                t_b, f_b = self.play_q.popleft()
                g = self._gain_block(t_b, t_b + len(f_b) / self.sr)
                out = np.ascontiguousarray(f_b * g[:, None], dtype=np.float32)
                if self.out_dev:
                    self.out_dev.write(out)
                if self.wav_f:
                    pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16)
                    self.wav_f.writeframes(pcm.tobytes())
        if self.out_dev:
            self.out_dev.stop()
            self.out_dev.close()
        if self.sd_cap:
            if hasattr(self, "_stop_cap"):
                self._stop_cap.set()
            self.sd_cap.stop()
            self.sd_cap.close()

    def _toggle(self):
        self.enabled = not self.enabled
        print(f"[{'开' if self.enabled else '旁路'}]")


def main():
    # 单实例锁：端口被占说明已有实例在跑，直接退出（双实例=双份声音）
    # --no-lock 仅供离线 --file 评测（与线上实例并存跑回放）
    import socket
    _lock = socket.socket()
    if "--no-lock" not in sys.argv:
        try:
            _lock.bind(("127.0.0.1", 8479))
        except OSError:
            print("已有 rt_suppress 实例在运行，退出")
            sys.exit(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-device", default=None, help="回采端点关键词（默认 CABLE）")
    ap.add_argument("--play-device", default=None, help="播放端点关键词（默认 GAMING）")
    ap.add_argument("--file", default=None, help="离线回放 wav（验证链路用）")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--null", action="store_true", help="不播放到设备")
    ap.add_argument("--write", default=None, help="把处理输出写成 wav")
    ap.add_argument("--log", default=None, help="事件判定 jsonl")
    ap.add_argument("--no-frame", action="store_true",
                    help="关闭帧级增益路径（纯事件门控，对比用）")
    ap.add_argument("--frame-pow", type=float, default=2.0,
                    help="帧级增益幂塑形指数（站立/无键态，越大越狠误伤越高）")
    ap.add_argument("--frame-pow-pose", type=float, default=3.0,
                    help="帧级增益幂塑形指数（键盘移动姿态背书时，默认 3）")
    ap.add_argument("--no-lock", action="store_true",
                    help=argparse.SUPPRESS)  # 离线 --file 评测跳过单实例锁
    args = ap.parse_args()
    cfg = json.load(open(ROOT / "config.json", encoding="utf-8"))
    args.block = cfg["block_size"]
    if args.file and not args.log:
        args.log = str(Path(args.file).parent / "rt_events.jsonl")
    rt = RtSuppress(cfg, args)
    src = None
    if args.file:
        src = FileSource(args.file, sr=48000, block=cfg["block_size"], speed=args.speed)
    try:
        rt.run(src)
    finally:
        if src:
            src.close()
        if rt.wav_f:
            rt.wav_f.close()
        if rt.rj:
            rt.rj.close()


if __name__ == "__main__":
    main()

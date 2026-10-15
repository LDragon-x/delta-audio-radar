"""语音报点识别：素材录制时口头报"铁板/草地/跑步"等标记，自动对齐时间轴。

用法：
  python tool/detect_marks.py <会话目录> [--model tiny]

产物：会话目录/voice_marks.jsonl，每行 {"t": 秒, "text": 原文, "labels": [命中的标签]}
喂数据时按标签把音频切桶（姿态/地形）。

关键词表（可扩展，用户报点时尽量用这些词）：
  地形：水泥 铁板 木头 木板 草地 土 沙 水 雪地 瓷砖
  姿态：走 走路 跑 跑步 跳 跳落 蹲 蹲走 爬 趴
  其它：换（换地形了）
"""
import argparse
import json
import re
import sys
from pathlib import Path

KEYWORDS = {
    "水泥": ["水泥"], "铁板": ["铁板", "金属", "钢板"], "木头": ["木头", "木板", "木箱"],
    "草地": ["草地", "草"], "沙地": ["沙", "土"], "水": ["水", "积水", "泥"],
    "雪地": ["雪"],
    "走": ["走路", "行走", "走"], "跑": ["跑步", "跑", "冲刺"], "跳": ["跳"],
    "蹲": ["蹲"], "爬": ["爬", "趴"],
}


def match_labels(text):
    labs = []
    for name, pats in KEYWORDS.items():
        for p in pats:
            if p in text:
                labs.append(name)
                break
    return labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="会话目录（含 session.wav）")
    ap.add_argument("--model", default="tiny", choices=["tiny", "base", "small"])
    ap.add_argument("--lang", default="zh")
    args = ap.parse_args()

    d = Path(args.session)
    wav = d / "session.wav"
    if not wav.exists():
        sys.exit(f"找不到 {wav}")

    from faster_whisper import WhisperModel
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(wav), language=args.lang, beam_size=1,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300),
        condition_on_previous_text=False)
    out = d / "voice_marks.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for seg in segments:
            text = re.sub(r"\s+", "", seg.text)
            labs = match_labels(text)
            if labs or len(text) >= 2:   # 有命中或成句的都存，便于人工复核
                f.write(json.dumps({"t": round(seg.start, 2),
                                    "end": round(seg.end, 2),
                                    "text": text, "labels": labs},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(f"识别 {n} 段语音 → {out}")


if __name__ == "__main__":
    main()

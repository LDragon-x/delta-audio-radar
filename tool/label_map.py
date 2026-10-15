"""标注值 → 规范声音类。标注页新写法（大类:具体）、旧写法（enemy_footstep）、
用户随手写的中文（局内播报）全部统一映射到规范类，learn/extract 共用一套口径。

规范类：footstep / bot_footstep / gunshot / self_footstep / self_gunshot /
broadcast / loot / bot_vocal / melee / skill / rescue / other
"""
LEGACY = {
    "enemy_footstep": "footstep", "bot_footstep": "bot_footstep",
    "gunshot": "gunshot", "enemy_gunshot": "gunshot", "人机枪声": "gunshot",
    "self_footstep": "self_footstep", "self_gunshot": "self_gunshot",
    "局内播报": "broadcast", "搜物质的声音": "loot", "搜物资": "loot",
    "人机死亡鬼叫": "bot_vocal", "人机说话": "bot_vocal",
    "切刀砍人机": "melee", "技能声": "skill",
    "死人威龙开局丢炸弹": "skill", "威龙丢炸弹": "skill",
    "敌人救援": "rescue", "受伤声音": "other",
}


def class_of(lab):
    """标注值（可能带 :具体描述）→ 规范类名，认不出返回 None。"""
    lab = (lab or "").strip()
    if not lab:
        return None
    if lab in LEGACY:
        return LEGACY[lab]
    cat, _, det = lab.partition(":")
    det = det.strip()
    has_gun = any(k in det for k in ("枪", "雷", "炸", "炮"))
    has_step = any(k in det for k in ("步", "脚", "走", "跑", "跳"))
    if cat == "enemy":
        if has_gun:
            return "gunshot"
        if any(k in det for k in ("技能", "炸弹", "烟", "火", "喷")):
            return "skill"
        if any(k in det for k in ("门", "弹", "药", "杂", "水")):
            return "other"
        return "footstep"
    if cat == "bot":
        if has_gun:
            return "gunshot"
        if any(k in det for k in ("说", "叫")):
            return "bot_vocal"
        if any(k in det for k in ("门", "杂")):
            return "other"
        return "bot_footstep"
    if cat == "self":
        if has_gun:
            return "self_gunshot"
        if any(k in det for k in ("技能", "炸弹", "烟", "火")):
            return "skill"
        return "self_footstep"
    if cat == "rescue":
        return "rescue"
    if cat == "skill":
        return "skill"
    if cat == "env":
        if "播报" in det:
            return "broadcast"
        if "搜" in det:
            return "loot"
        if any(k in det for k in ("鬼叫", "说话", "叫")):
            return "bot_vocal"
        if "刀" in det:
            return "melee"
        if any(k in det for k in ("技能", "炸弹", "烟", "火", "喷气")):
            return "skill"
        if "救援" in det:
            return "rescue"
        return None
    if cat == "other":
        return "other"
    return None

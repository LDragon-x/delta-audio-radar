"""容器占卜师：容器类型 × 格子形状 × 加载圈数 → 物品推断与决策建议。

三角洲烽火地带的核心预判链：
  容器决定品类 → 形状锁定物品 → 圈数确认品质 → 价格库给估值 → 拿/不拿/中断

用法：
  python tool/container_oracle.py --container 大保险箱 --shape 3x3
  python tool/container_oracle.py --spinner 2
  python tool/container_oracle.py --list-containers
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKET = ROOT / "data" / "market"


def load_all():
    intel = json.loads((MARKET / "containers.json").read_text(encoding="utf-8"))
    snaps = sorted(MARKET.glob("items_*.json"))
    snap = json.loads(snaps[-1].read_text(encoding="utf-8"))
    prices = {x["name"]: x["prices"].get("current") or 0 for x in snap["items"]}
    return intel, prices


def price_of(prices, name):
    if name in prices:
        return prices[name]
    # 模糊匹配：名称包含
    for k, v in prices.items():
        if name in k or k in name:
            return v
    return None


def spinner_to_rarity(intel, revs):
    table = intel["_search_time"]["spinner_revolutions"]
    key = str(revs)
    if key in table:
        return table[key]
    # 就近匹配
    best = min(table, key=lambda k: abs(float(k) - revs))
    return f"{table[best]}（就近匹配 {best} 圈）"


def advise(intel, prices, container, shape, revs=None):
    cname = container
    # 锁定组合优先
    combo = intel["_locked_combos"].get(f"{cname}:{shape}")
    fraud = intel["_fraud_shapes"].get(f"{cname}:{shape}")
    cinfo = intel["_containers"].get(cname)

    print(f"== 容器[{cname}] 形状[{shape}]"
          f"{' 圈数[' + str(revs) + ']' if revs else ''} ==")
    if cinfo:
        print(f"  爆率[{cinfo['tier']}] 产出品类: {', '.join(cinfo['drops'])}"
              + (f"  ({cinfo['note']})" if cinfo.get("note") else ""))
    if revs is not None:
        print(f"  品质推断: {spinner_to_rarity(intel, revs)}")

    if fraud:
        p = price_of(prices, fraud)
        print(f"\n  ⚠ 诈骗场景: 此形状在此容器大概率是 [{fraud}]"
              + (f"（约 {p:,} 哈夫币）" if p else "（无价）")
              + " → 建议中断搜索")
        return

    if combo:
        cands = combo.split("/")
        print(f"\n  ★ 锁定高价值: {combo}")
        for c in cands:
            p = price_of(prices, c)
            print(f"    {c:<12} {p:>12,} 哈夫币" if p else f"    {c:<12} (价格库未收录)")
        return

    # 通用形状候选 + 容器品类过滤
    shape_cands = intel["_shape_map"].get(shape, [])
    if cinfo:
        drops = " ".join(cinfo["drops"])
        shape_cands = [c for c in shape_cands
                       if price_of(prices, c) is not None] or shape_cands
    if shape_cands:
        print(f"\n  形状候选（{len(shape_cands)} 个）:")
        for c in shape_cands:
            p = price_of(prices, c)
            mark = "★" if p and p >= 1_000_000 else ("·" if p and p >= 100_000 else " ")
            print(f"  {mark} {c:<14} {p:>12,}" if p else f"    {c:<14} (无价)")
    else:
        print("\n  该形状无大红记录 → 大概率蓝白货，建议中断搜索")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container")
    ap.add_argument("--shape")
    ap.add_argument("--spinner", type=float)
    ap.add_argument("--list-containers", action="store_true")
    args = ap.parse_args()

    intel, prices = load_all()
    if args.list_containers:
        for name, info in intel["_containers"].items():
            print(f"  [{info['tier']}] {name}: {', '.join(info['drops'])}")
        return
    if not args.container or not args.shape:
        ap.print_help()
        return
    advise(intel, prices, args.container, args.shape, args.spinner)


if __name__ == "__main__":
    main()

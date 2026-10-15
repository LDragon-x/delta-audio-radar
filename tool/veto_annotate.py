"""否决规则标注器：给物价快照里的每件物品打上"值不值得看"的标。

产品哲学（用户定调）：白品直接否决、绿品大概率否决、无意义品类踢掉——
垃圾不上桌。被否决的物品在局内 overlay 里直接跳过（连识别都不做）。

用法：python tool/veto_annotate.py [快照日期]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKET = ROOT / "data" / "market"


def load_latest_snapshot(day=None):
    # 优先用单格价值表（含 dgjz/wh/grade），没有则退回价格快照
    cell = sorted(MARKET.glob("cellvalue_*.json"))
    if cell:
        items = json.loads(cell[-1].read_text(encoding="utf-8"))
        for x in items:
            x["prices"] = {"current": x.get("price")}
        return {"date": cell[-1].stem.split("_")[-1], "items": items}
    snaps = sorted(MARKET.glob("items_*.json"))
    if day:
        snaps = [s for s in snaps if day in s.name]
    return json.loads(snaps[-1].read_text(encoding="utf-8"))


def decide(item, rules):
    name = item["name"]
    price = item["prices"].get("current") or 0
    cell_v = item.get("dgjz")

    # 例外优先于黑名单
    for keep in rules.get("name_patterns_keep", []):
        if keep in name:
            return False, f"例外保留({keep})"
    for pat in rules.get("name_patterns_veto", []):
        if pat in name:
            return True, f"品类黑名单({pat})"

    # 单位格价值否决（最本质：4格占位垃圾被这条处决）
    if cell_v is not None and cell_v < rules.get("cell_value_below", 0):
        return True, f"否决: {cell_v}/格 < {rules['cell_value_below']}/格"

    if price < rules.get("price_below", 0):
        return True, f"否决: {price} < {rules['price_below']}"

    return False, "保留"


def main():
    snap = load_latest_snapshot(sys.argv[1] if len(sys.argv) > 1 else None)
    rules = json.loads((MARKET / "rules.json").read_text(encoding="utf-8"))["veto"]

    kept, vetoed = [], []
    for it in snap["items"]:
        v, reason = decide(it, rules)
        it["veto"], it["veto_reason"] = v, reason
        (vetoed if v else kept).append(it)

    out = MARKET / f"items_annotated_{snap['date']}.json"
    snap["veto_rules"] = rules
    snap["kept_count"] = len(kept)
    snap["vetoed_count"] = len(vetoed)
    snap["items"] = sorted(snap["items"], key=lambda x: (x["veto"],
                          -(x["prices"].get("current") or 0)))
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"快照 {snap['date']}：{len(snap['items'])} 件 → 否决 {len(vetoed)}，保留 {len(kept)}")
    print("\n否决原因分布：")
    from collections import Counter
    for r, c in Counter(x["veto_reason"].split(":")[0].split("(")[0]
                        for x in vetoed).most_common():
        print(f"  {r}: {c}")
    print("\n保留价值 Top 10（按单位格价值）:")
    for x in sorted(kept, key=lambda x: -(x.get("dgjz") or x["prices"].get("current") or 0))[:10]:
        p = x["prices"].get("current") or 0
        cat = x.get("category") or f"{x.get('grade', '?')}级"
        cv = f"{x.get('dgjz'):,}/格" if x.get("dgjz") else ""
        print(f"  {x['name']:<16} {cat:<8} {p:>12,}  {cv}")
    print(f"\n已存 {out}")


if __name__ == "__main__":
    main()

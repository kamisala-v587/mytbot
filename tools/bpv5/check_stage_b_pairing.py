"""Validate Stage B demo-pairing rules: sample 3 pairs per source, dump first/last head frames + bilingual text.

Rules from: code/conclusion/bpv5/Stage B 示教配对分析.md
  L0: same repo (ManipArena: also same sim/real prefix)
  L1: same repo + exact normalize_task_text (HiFi-UMI: exact text across chunks)
  L2/DOM: show that exact full-text buckets are mostly singletons (no default pair)

usage:
  cd /vla/workspace/my_tbot
  python tools/bpv5/check_stage_b_pairing.py [--out DIR] [--seed 0] [--pairs 3]
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.policies.BPV5QFormer import data80d as D  # noqa: E402

# (source, tier, pairing note)
SOURCES = [
    ("abc", "L0", "同 repo"),
    ("agibot", "L0", "同 repo"),
    ("worldengine", "L0", "同 repo"),
    ("maniparena", "L0", "同 repo；sim/real 不互配"),
    ("interndata", "L0", "同 repo"),
    ("robotwin", "L0", "同 repo（勿用精确句面）"),
    ("galaxea", "L0", "同 repo（忽略逐步句面）"),
    ("openneo", "L1", "同 repo + 精确任务句"),
    ("hiw", "L1", "同 repo + 精确任务句"),
    ("icra_wbc", "L1", "同 repo + 精确任务句（含品牌）"),
    ("hifi_umi", "L1", "跨 chunk 精确任务句"),
    ("dom", "L2", "精确全文不可行；本脚本只展示 3 条样本与 singleton 现象"),
]


def raw_task(rel: str, ep: int):
    return D.episode_tasks(rel).get(int(ep))


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in text)


# Manual Chinese glosses for sampled English instructions (pair-check review). Extend when re-sampling.
EN_TO_ZH: dict[str, str] = {
    "put the phone into the phone case": "把手机放进手机壳里",
    "organize the makeup": "整理化妆品",
    "place and organize the paper towels onto the shelf": "把纸巾放到架子上并整理好",
    "Place items from the meeting room table into the storage box. | The robot faces the table, with the remote control and pen randomly placed on the tabletop.": "把会议室桌上的物品放进收纳盒。| 机器人面向桌子，遥控器与笔随机放在桌面上。",
    "Place the desktop items into the felt bag. | The robot is in front of the table, with the felt package and the object to be operated on both placed on the table surface.": "把桌面上的物品放进毛毡袋。| 机器人在桌前，毛毡包装与待操作物体都在桌面上。",
    "Packing in e-commerce | The robot stands in front of the table, with a cardboard box, air column wrap, and items arranged in sequence on the table.": "电商打包 | 机器人站在桌前，纸箱、气柱膜与物品依次摆在桌上。",
    "sort-kitchen-utensils": "整理厨房用具",
    "shopping-basket-loading": "往购物篮里装东西",
    "open-sealed-box-and-take-something-out": "打开密封盒并取出物品",
    "Press the green, pink, and yellow buttons in order.": "按顺序按下绿色、粉色和黄色按钮。",
    "Press the pink, yellow, and blue buttons in order.": "按顺序按下粉色、黄色和蓝色按钮。",
    "Pour the water from the kettle into the cup.": "把水壶里的水倒进杯子。",
    "Pick up the shoes": "拿起鞋子",
    "Pick up the hat with right arm": "用右臂拿起帽子",
    "Collect white school glue with orange cap, yellow-red glue with white cap and yellow glue with nozzle in the blue basket.": "把橙盖白胶、白盖黄红胶和带喷嘴黄胶收进蓝色篮子。",
    "Collect yellow glue with nozzle, white glue with green cap and yellow-red glue with white cap in the blue basket.": "把带喷嘴黄胶、绿盖白胶和白盖黄红胶收进蓝色篮子。",
    "Pick up the lemon with left arm": "用左臂拿起柠檬",
    "First move the rounded metallic can to the rectangular storage box with grooves, then move the can with rounded cylindrical shape into the rectangular storage box with grooves.": "先把圆形金属罐移到带凹槽的长方形收纳盒，再把圆筒形罐子移进同一收纳盒。",
    "Lift the palm-sized drink can, put it in the plasticbox for carrying items, then handle the drink can with smooth red finish similarly.": "拿起掌心大小的饮料罐放进装物塑料盒，再对光滑红色饮料罐做同样操作。",
    "Stack the medium-sized bowl for holding food carefully over the bowl with shiny texture.": "小心地把中号盛食碗叠到有光泽的碗上面。",
    "Hold the blue bowl and stack the medium-sized bowl for holding food neatly atop.": "扶住蓝色碗，再把中号盛食碗整齐叠在上面。",
    "Grab the oval-shaped mouse and drop it on Green mat.": "抓住椭圆形鼠标，放到绿色垫子上。",
    "Use the left arm to pick the gray mouse with smooth surface up and set it on the Yellow colored mat.": "用左臂拿起表面光滑的灰色鼠标，放到黄色垫子上。",
    "pour out the leftover food null qualified": "倒掉剩饭剩菜（null / qualified）",
    "brush pickup": "拿起刷子",
    "assemble the battery": "组装电池",
    "plug usb light particle": "插入 USB 灯颗粒/配件",
    "hang keys on a hook": "把钥匙挂到挂钩上",
    "picking trash to rubbish bin": "把垃圾捡进垃圾桶",
    "clean up the room": "清理房间",
    "Move in front of the shelf. Grasp the Mirinda. Hold the item and walk to the cart. Place the item into the cart": "走到货架前，抓住美年达，拿着物品走到购物车旁，放进购物车。",
    "Move in front of the shelf. Grasp the Pepsi. Hold the item and walk to the cart. Place the item into the cart": "走到货架前，抓住百事可乐，拿着物品走到购物车旁，放进购物车。",
    "Move in front of the shelf. Grasp the Frozen Pear Drink. Hold the item and walk to the cart. Place the item into the cart": "走到货架前，抓住冻梨饮料，拿着物品走到购物车旁，放进购物车。",
    "Put the cup on the cup holder.": "把杯子放到杯架上。",
    "Close the meeting room blinds.": "关上会议室百叶窗。",
    "Open the bathtub drain.": "打开浴缸排水口。",
}


def translate_en_to_zh(eng: str) -> str:
    """Chinese gloss for review when the dataset has no Chinese side."""
    eng = (eng or "").strip()
    if not eng:
        return ""
    if eng in EN_TO_ZH:
        return EN_TO_ZH[eng]
    if eng.startswith(("long_horizon;", "pick;", "place;")):
        return (
            eng.replace("long_horizon", "长时程", 1)
            .replace("objects:", "物体：")
            .replace("containers:", "容器：")
            .replace("pick;", "抓取；", 1)
            .replace("place;", "放置；", 1)
        )
    # kebab task ids
    if all(c.isalnum() or c in "-_" for c in eng) and ("-" in eng or "_" in eng):
        return eng.replace("-", " ").replace("_", " ")
    return eng  # unknown: keep EN; add to EN_TO_ZH when reviewing new samples


def bilingual(rel: str, ep: int) -> dict:
    raw = raw_task(rel, ep)
    eng = D.task_text(rel, ep)
    zh = ""
    zh_source = "none"
    # try recover Chinese side from raw string / list / galaxea bilingual
    candidates = []
    if isinstance(raw, list):
        candidates = [str(x) for x in raw]
    elif isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, dict):
        candidates = [json.dumps(raw, ensure_ascii=False)]
    for s in candidates:
        if _has_cjk(s):
            # keep Chinese-heavy segments
            parts = []
            for sep in ("\n", " | ", "|", " / ", "/", "@"):
                if sep in s:
                    parts = [p.strip() for p in s.split(sep) if p.strip()]
                    break
            if not parts:
                parts = [s]
            zh_parts = [p for p in parts if _has_cjk(p)]
            if zh_parts:
                zh = zh_parts[0]
            else:
                zh = s
            zh_source = "dataset"
            break
    if not zh:
        zh = translate_en_to_zh(eng)
        zh_source = "translate"
    return {
        "raw": raw if not isinstance(raw, (dict, list)) else raw,
        "en": eng,
        "zh": zh,
        "zh_source": zh_source,
    }


def manips_domain(rel: str) -> str:
    # maniparena/real/... or maniparena/sim/...
    parts = rel.split("/")
    return parts[1] if len(parts) > 1 and parts[1] in {"sim", "real"} else "unknown"


def list_eps(rel: str, limit_scan: int | None = None) -> list[int]:
    eps = sorted(D.episodes(rel))
    if limit_scan is not None:
        eps = eps[:limit_scan]
    return eps


def pair_key_l0(rel: str, ep: int) -> str:
    if rel.startswith("maniparena/"):
        return f"{rel}||{manips_domain(rel)}"
    return rel


def pair_key_l1(rel: str, ep: int) -> str:
    text = D.task_text(rel, ep)
    if rel.startswith("hifi_umi/"):
        return f"hifi_umi||{text}"
    return f"{rel}||{text}"


def sample_repos(source: str, rng: random.Random, k: int = 24) -> list[str]:
    rels = D.repos([source])
    if len(rels) <= k:
        return list(rels)
    return rng.sample(rels, k)


def build_l0_pairs(source: str, rng: random.Random, n_pairs: int) -> list[dict]:
    pairs = []
    rels = sample_repos(source, rng, k=40)
    rng.shuffle(rels)
    for rel in rels:
        if source == "abc" and "/val__" in rel:
            # doc: drop tiny val__* repos when possible
            eps = list_eps(rel)
            if len(eps) <= 3:
                continue
        try:
            eps = list_eps(rel)
        except Exception:
            continue
        if len(eps) < 4:
            continue
        if source == "maniparena":
            # keep domain encoded in key; still same repo so ok
            pass
        a, b = rng.sample(eps, 2)
        pairs.append(
            {
                "source": source,
                "tier": "L0",
                "pair_key": pair_key_l0(rel, a),
                "query": {"rel": rel, "ep": int(a)},
                "demo": {"rel": rel, "ep": int(b)},
                "bucket_size_est": len(eps),
            }
        )
        if len(pairs) >= n_pairs:
            break
    return pairs


def build_l1_pairs(source: str, rng: random.Random, n_pairs: int) -> list[dict]:
    """Build exact-text buckets from a sampled subset, then draw pairs with bucket>3 when possible."""
    buckets: dict[str, list[tuple[str, int]]] = defaultdict(list)
    if source == "hifi_umi":
        rels = sample_repos(source, rng, k=30)
        for rel in rels:
            try:
                eps = list_eps(rel, limit_scan=80)
            except Exception:
                continue
            for ep in eps:
                buckets[pair_key_l1(rel, ep)].append((rel, int(ep)))
    elif source == "hiw":
        rel = "hiw/main"
        # scan more episodes to fill 10 task buckets
        eps = list_eps(rel, limit_scan=2000)
        # subsample for speed
        if len(eps) > 800:
            eps = rng.sample(eps, 800)
        for ep in eps:
            buckets[pair_key_l1(rel, ep)].append((rel, int(ep)))
    elif source == "icra_wbc":
        for rel in D.repos([source]):
            eps = list_eps(rel, limit_scan=400)
            if len(eps) > 300:
                eps = rng.sample(eps, 300)
            for ep in eps:
                buckets[pair_key_l1(rel, ep)].append((rel, int(ep)))
    else:  # openneo
        rels = sample_repos(source, rng, k=min(7, len(D.repos([source]))))
        for rel in rels:
            eps = list_eps(rel, limit_scan=3000)
            if len(eps) > 1500:
                eps = rng.sample(eps, 1500)
            for ep in eps:
                buckets[pair_key_l1(rel, ep)].append((rel, int(ep)))

    usable = [(k, v) for k, v in buckets.items() if len(v) >= 4]
    if not usable:
        usable = [(k, v) for k, v in buckets.items() if len(v) >= 2]
    rng.shuffle(usable)
    pairs = []
    for key, items in usable:
        a, b = rng.sample(items, 2)
        pairs.append(
            {
                "source": source,
                "tier": "L1",
                "pair_key": key,
                "query": {"rel": a[0], "ep": a[1]},
                "demo": {"rel": b[0], "ep": b[1]},
                "bucket_size_est": len(items),
                "same_text": D.task_text(a[0], a[1]) == D.task_text(b[0], b[1]),
            }
        )
        if len(pairs) >= n_pairs:
            break
    return pairs


def build_dom_samples(rng: random.Random, n: int = 3) -> tuple[list[dict], dict]:
    """DOM: measure singleton rate on a sample; return n unpaired examples (exact key)."""
    rel = "dom"
    eps = list_eps(rel, limit_scan=5000)
    sample = rng.sample(eps, min(2000, len(eps)))
    buckets: dict[str, list[int]] = defaultdict(list)
    for ep in sample:
        buckets[D.task_text(rel, ep)].append(int(ep))
    sizes = np.array([len(v) for v in buckets.values()])
    stats = {
        "sampled_episodes": len(sample),
        "exact_text_buckets": int(len(buckets)),
        "singleton_buckets": int((sizes == 1).sum()),
        "singleton_frac": float((sizes == 1).mean()),
        "median_bucket": float(np.median(sizes)),
        "max_bucket": int(sizes.max()),
        "buckets_ge_4": int((sizes >= 4).sum()),
    }
    # prefer showing near-duplicates that are still different exact keys if possible
    examples = []
    # group by task field only
    by_task: dict[str, list[int]] = defaultdict(list)
    for ep in sample[:800]:
        raw = raw_task(rel, ep)
        try:
            obj = json.loads(raw) if isinstance(raw, str) else raw
            t = str(obj.get("task")) if isinstance(obj, dict) else "unk"
        except Exception:
            t = "unk"
        by_task[t].append(int(ep))
    for t, eps_t in by_task.items():
        if len(eps_t) < 2:
            continue
        a, b = rng.sample(eps_t, 2)
        ta, tb = D.task_text(rel, a), D.task_text(rel, b)
        examples.append(
            {
                "source": "dom",
                "tier": "L2",
                "pair_key": "(exact full text — usually singleton)",
                "query": {"rel": rel, "ep": a},
                "demo": {"rel": rel, "ep": b},
                "same_coarse_task": t,
                "exact_text_equal": ta == tb,
                "bucket_size_est": len(buckets.get(ta, [])),
            }
        )
        if len(examples) >= n:
            break
    while len(examples) < n and sample:
        ep = int(rng.choice(sample))
        examples.append(
            {
                "source": "dom",
                "tier": "L2",
                "pair_key": D.task_text(rel, ep)[:120],
                "query": {"rel": rel, "ep": ep},
                "demo": {"rel": rel, "ep": ep},  # self — illustrates no partner
                "exact_text_equal": True,
                "bucket_size_est": len(buckets.get(D.task_text(rel, ep), [])),
                "note": "no alternate under exact key",
            }
        )
    return examples[:n], stats


def head_camera(rel: str) -> str | None:
    slots = D.camera_slots(rel)
    return slots[0]


def decode_ends(rel: str, ep: int, size: int = 256) -> tuple[np.ndarray, np.ndarray] | None:
    """Return first/last head-camera RGB uint8 HxWx3, or None."""
    from torchcodec.decoders import VideoDecoder
    from lerobot.transforms.utils import resize_with_pad
    import torch

    cam = head_camera(rel)
    if not cam:
        return None
    meta = D.episodes(rel)[ep]
    chunk, file, t0 = meta["videos"][cam]
    path = D.video_path(rel, cam, chunk, file)
    if not path.is_file():
        # softlink may point outside; still try
        pass
    length = int(meta["length"])
    fps = float(D.info(rel)["fps"])
    dec = VideoDecoder(str(path), seek_mode="exact", device="cpu")
    vfps, last = float(dec.metadata.average_fps), int(dec.metadata.num_frames) - 1
    idxs = [min(round((t0 + 0.0 / fps) * vfps), last), min(round((t0 + (length - 1) / fps) * vfps), last)]
    batch = dec.get_frames_at(indices=idxs).data.float() / 255.0  # [2,C,H,W]
    batch = resize_with_pad(batch, size, size).cpu()
    imgs = (batch.permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return imgs[0], imgs[1]


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
    except Exception:
        import imageio.v2 as imageio

        imageio.imwrite(path, rgb)


def b64_png(rgb: np.ndarray) -> str:
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/vla/workspace/code/conclusion/bpv5/assets/pair_check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    report = {"seed": args.seed, "pairs_per_source": args.pairs, "sources": []}
    html_sections = []

    for source, tier, rule in SOURCES:
        print(f"=== {source} ({tier}) ===", flush=True)
        if source == "dom":
            pairs, dom_stats = build_dom_samples(rng, args.pairs)
            src_rec = {"source": source, "tier": tier, "rule": rule, "dom_stats": dom_stats, "pairs": []}
            print("  DOM exact-text stats:", json.dumps(dom_stats), flush=True)
        elif tier == "L0":
            pairs = build_l0_pairs(source, rng, args.pairs)
            src_rec = {"source": source, "tier": tier, "rule": rule, "pairs": []}
        else:
            pairs = build_l1_pairs(source, rng, args.pairs)
            src_rec = {"source": source, "tier": tier, "rule": rule, "pairs": []}

        cards = []
        for i, p in enumerate(pairs):
            q, d = p["query"], p["demo"]
            qb, db = bilingual(q["rel"], q["ep"]), bilingual(d["rel"], d["ep"])
            entry = {
                **p,
                "query_text": qb,
                "demo_text": db,
                "frames": {},
                "ok_same_key_texts_match_rule": None,
            }
            # validity checks
            if tier == "L0":
                entry["ok"] = q["rel"] == d["rel"] and q["ep"] != d["ep"]
                if source == "maniparena":
                    entry["ok"] = entry["ok"] and manips_domain(q["rel"]) == manips_domain(d["rel"])
            elif tier == "L1":
                entry["ok"] = qb["en"] == db["en"] and not (q["rel"] == d["rel"] and q["ep"] == d["ep"])
            else:
                entry["ok"] = bool(p.get("exact_text_equal")) is False or p.get("note")
                entry["ok_note"] = "L2: exact equal would be desired for pair; often False under full text"

            # frames
            for role, item in (("query", q), ("demo", d)):
                try:
                    ends = decode_ends(item["rel"], item["ep"])
                except Exception as e:  # noqa: BLE001
                    entry["frames"][role] = {"error": f"{type(e).__name__}: {e}"}
                    print(f"  frame fail {source} {role} {item}: {e}", flush=True)
                    continue
                if ends is None:
                    entry["frames"][role] = {"error": "no head camera"}
                    continue
                first, last = ends
                stem = f"{source}_p{i}_{role}_ep{item['ep']}"
                fp = out / f"{stem}_first.jpg"
                lp = out / f"{stem}_last.jpg"
                save_png(fp, first)
                save_png(lp, last)
                entry["frames"][role] = {
                    "first": str(fp),
                    "last": str(lp),
                    "first_b64": b64_png(first),
                    "last_b64": b64_png(last),
                    "rel": item["rel"],
                    "ep": item["ep"],
                }
            src_rec["pairs"].append(entry)
            # strip huge b64 from json dump later
            print(
                f"  pair{i}: q={q['rel']}|{q['ep']}  d={d['rel']}|{d['ep']}  "
                f"en_q={qb['en'][:60]!r}  ok={entry.get('ok')}",
                flush=True,
            )

            def img_tag(role, which):
                fr = entry["frames"].get(role) or {}
                if "error" in fr:
                    return f"<div class='err'>{role}/{which}: {fr['error']}</div>"
                return f"<img src='data:image/jpeg;base64,{fr[which + '_b64']}' alt='{role} {which}'/>"

            cards.append(
                f"""
<div class="pair">
  <h3>#{i+1} · bucket≈{p.get('bucket_size_est')} · ok={entry.get('ok')}</h3>
  <div class="meta"><b>pair_key</b>: <code>{p.get('pair_key')}</code></div>
  <div class="grid">
    <div>
      <h4>Query ep{q['ep']}</h4>
      <div class="txt"><b>EN</b>: {qb['en']}</div>
      <div class="txt"><b>ZH</b>: {qb['zh']}</div>
      <div class="frames">{img_tag('query','first')}{img_tag('query','last')}</div>
      <div class="cap">first / last · {q['rel']}</div>
    </div>
    <div>
      <h4>Demo ep{d['ep']}</h4>
      <div class="txt"><b>EN</b>: {db['en']}</div>
      <div class="txt"><b>ZH</b>: {db['zh']}</div>
      <div class="frames">{img_tag('demo','first')}{img_tag('demo','last')}</div>
      <div class="cap">first / last · {d['rel']}</div>
    </div>
  </div>
</div>"""
            )

        # drop b64 from json
        slim_pairs = []
        for e in src_rec["pairs"]:
            e2 = {k: v for k, v in e.items() if k != "frames"}
            fr = {}
            for role, fv in (e.get("frames") or {}).items():
                fr[role] = {k: v for k, v in fv.items() if not k.endswith("_b64")}
            e2["frames"] = fr
            slim_pairs.append(e2)
        src_rec["pairs"] = slim_pairs
        report["sources"].append(src_rec)

        html_sections.append(
            f"<section><h2>{source} · {tier}</h2><p class='rule'>{rule}</p>"
            + (f"<pre>{json.dumps(src_rec.get('dom_stats'), indent=2, ensure_ascii=False)}</pre>" if source == "dom" else "")
            + "".join(cards)
            + "</section>"
        )

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>Stage B 示教配对抽检</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:#f6f7f9;color:#111}}
h1{{margin:0 0 8px}} h2{{margin-top:36px;border-bottom:1px solid #ccc;padding-bottom:6px}}
.rule{{color:#444;margin:4px 0 16px}}
.pair{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px;margin:14px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.frames{{display:flex;gap:8px;flex-wrap:wrap}}
.frames img{{width:240px;height:240px;object-fit:cover;border-radius:4px;background:#222}}
.txt{{font-size:13px;margin:4px 0;line-height:1.35;word-break:break-word}}
.cap,.meta{{font-size:12px;color:#666}}
.err{{color:#a00;font-size:12px}}
code{{font-size:12px}}
pre{{background:#eee;padding:8px;overflow:auto}}
</style></head><body>
<h1>Stage B 示教配对规则抽检</h1>
<p>种子={args.seed}；每源 {args.pairs} 对；规则见《Stage B 示教配对分析》。首末帧=头相机。</p>
{"".join(html_sections)}
</body></html>"""
    html_path = out / "index.html"
    html_path.write_text(html, encoding="utf-8")
    print("wrote", html_path)
    print("wrote", out / "report.json")


if __name__ == "__main__":
    main()

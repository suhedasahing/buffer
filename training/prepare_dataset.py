"""YOLO formatindaki birden cok veri kaynagini tek sinifa harmanlar, kalite filtreler ve train/val/test ayirir.

Kullanim:
    python training/prepare_dataset.py --config training/weapon_sources.json --out training/data/weapon_v2 --class-name weapon

Cikti:
    <out>/images/{train,val,test}
    <out>/labels/{train,val,test}
    <out>/data.yaml
    <out>/report.json
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve(root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


def parse_label(text: str, class_map: dict, min_box: float):
    """Etiket satirini cozumler. (cls, cx, cy, w, h) veya None doner."""
    parts = text.split()
    if len(parts) != 5:
        return None
    try:
        cls = int(float(parts[0]))
        vals = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    if cls not in class_map:
        return None
    cx, cy, w, h = vals
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
        return None
    if w < min_box or h < min_box:
        return None
    return (class_map[cls], cx, cy, w, h)


def find_images(img_dir: Path):
    return sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--class-name", default="weapon")
    ap.add_argument("--min-box", type=float, default=0.002)
    ap.add_argument("--keep-empty", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    report = {"class": args.class_name, "sources": [], "totals": {"train": 0, "val": 0, "test": 0}}

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    for src in cfg["sources"]:
        name = src["name"]
        root = Path(src["path"])
        if not root.is_dir():
            print(f"[SKIP] {name}: '{src['path']}' yok")
            continue
        class_map = {int(k): int(v) for k, v in src["class_map"].items()}
        src_report = {"name": name, "images": 0, "boxes": 0, "classes": {}, "dropped_empty": 0, "dropped_bad": 0, "splits": {}}

        splits_from_dirs = {s.get("split", s["dir"].lower()): s["dir"] for s in src.get("dirs", [])}

        items = []
        for raw_split, d in splits_from_dirs.items():
            img_dir = resolve(root, d) / "images"
            lab_dir = resolve(root, d) / "labels"
            if not img_dir.is_dir():
                print(f"[WARN] {name}: '{img_dir}' yok")
                continue
            for img_p in find_images(img_dir):
                lbl_p = lab_dir / (img_p.stem + ".txt")
                if not lbl_p.is_file():
                    continue
                items.append((raw_split, img_p, lbl_p))

        resplit = src.get("resplit")
        if resplit:
            random.Random(resplit.get("seed", 42)).shuffle(items)
            fracs = resplit.get("frac", {})
            n = len(items)
            n_train = int(n * fracs.get("train", 0.8))
            n_val = int(n * fracs.get("val", 0.1))
            assignments = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
            items = [(assignments[i], img_p, lbl_p) for i, (_, img_p, lbl_p) in enumerate(items)]

        for raw_split, img_p, lbl_p in items:
            out_split = src.get("override", {}).get(raw_split, raw_split)
            out_split = "val" if out_split == "valid" else out_split
            if out_split not in ("train", "val", "test"):
                continue
            with open(lbl_p, encoding="utf-8", errors="replace") as f:
                text = f.read()
            boxes = []
            if text.strip():
                for line in text.splitlines():
                    b = parse_label(line.strip(), class_map, args.min_box)
                    if b is None:
                        src_report["dropped_bad"] += 1
                    else:
                        boxes.append(b)
            if not boxes and not args.keep_empty:
                src_report["dropped_empty"] += 1
                continue
            dst_stem = f"{name}__{img_p.stem}"
            dst_img = out / "images" / out_split / (dst_stem + img_p.suffix.lower())
            shutil.copy2(img_p, dst_img)
            with open(out / "labels" / out_split / f"{dst_stem}.txt", "w", encoding="utf-8") as f:
                for b in boxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
            src_report["images"] += 1
            src_report["boxes"] += len(boxes)
            src_report["splits"][out_split] = src_report["splits"].get(out_split, 0) + 1
            for b in boxes:
                src_report["classes"][b[0]] = src_report["classes"].get(b[0], 0) + 1

        report["sources"].append(src_report)
        for s, v in src_report["splits"].items():
            report["totals"][s] += v

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\n"
        "names:\n  0: " + args.class_name + "\n",
        encoding="utf-8",
    )
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
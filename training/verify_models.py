"""Inen models/weapon_v2.pt ve models/cigarette_v2.pt'i bircok ornek kare uzerinde yukler ve dogrular.

Kullanim:
    python training/verify_models.py                     # uc model video/snap gibi seyler yoksa tek imaj
    python training/verify_models.py --img gun.gif       # (img/klasor/snapshot verilebilir)

Cikti: her model icin sinif adlari + her karedeki en yuksek gaven tespit (label conf).
Olasi sorunlar: model yuklenmiyorsa / sinif adi eslesmiyorsa bu script yakalar.
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.detector import level_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="")
    ap.add_argument("--weapon", default=str(ROOT / "models" / "weapon_v2.pt"))
    ap.add_argument("--cig", default=str(ROOT / "models" / "cigarette_v2.pt"))
    args = ap.parse_args()

    src = args.img or str(ROOT / "Gun-Detect-1" / "train" / "images")
    p = Path(src)
    assert p.exists(), f"kaynak yok: {p}"

    imgs = sorted({x for x in p.rglob("*") if x.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".mp4")})
    if not imgs:
        p = Path(src)
        if p.is_file():
            imgs = [p]
    if not imgs:
        print("ornek goruntu/video bulunamadi, --img ile verin")
        return 1
    imgs = imgs[:6]

    from ultralytics import YOLO

    checks = [("weapon", ROOT / "models" / "weapon.pt"), ("weapon_v2", args.weapon),
              ("cigarette_v2", args.cig)]
    models = []
    for tag, path in checks:
        if not Path(path).exists():
            print(f"[SKIP] {tag}: {path} yok")
            continue
        m = YOLO(str(path))
        models.append((tag, m))
        print(f"[{tag}] classes={list(m.names.values())} modelyolu={path}")

    ok = True
    for img_p in imgs:
        frame = cv2.imread(str(img_p)) if img_p.suffix.lower() != ".mp4" else None
        if frame is None:
            continue
        for tag, m in models:
            res = m(frame, conf=0.05, imgsz=config.IMGSZ, verbose=False)[0]
            for b in res.boxes:
                name = m.names[int(b.cls)]
                lvl = level_of(name)
                if lvl:
                    print(f"[{tag}] {img_p.name}: {name} conf={float(b.conf):.2f} -> level={lvl}")
                    break
            else:
                pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
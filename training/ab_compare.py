"""Webcam A/B: eski vs yeni modelleri ayni karelerde karsilastirir.

Kullanim (proje kokunden):
    python training/ab_compare.py [--seconds 8] [--device 0]

Cikti: her kare icin eski/yeni max gaven + eklgelerin (esik gecen kare sayisi).
Eskikler:
    weapon(.pt eski) vs weapon_v2, cigarette eski vs cigarette_v2.
Amac: modeli DEGISTIRMEDEN webcam degerlerine bakarak karar vermek.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402


def max_conf(results):
    return max([float(b.conf) for b in results[0].boxes] or [0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=8)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    pairs = [
        ("weapon", ROOT / "models" / "weapon.pt", ROOT / "models" / "weapon_v2.pt", config.CONF, config.IMGSZ),
        ("cigarette", ROOT / "models" / "cigarette.pt", ROOT / "models" / "cigarette_v2.pt", config.GREEN_CONF, config.GREEN_IMGSZ),
    ]
    models = []
    for tag, old_p, new_p, thr, sz in pairs:
        if not Path(old_p).exists() or not Path(new_p).exists():
            print(f"[SKIP] {tag}: model eksik")
            continue
        models.append((tag, YOLO(str(old_p)), YOLO(str(new_p)), thr, sz))

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print("kamera acilamadi")
        return 1
    n = 0
    print(f"{'frame':>5} ", end="")
    for tag, *_ in models:
        print(f"| {tag:<11} eski | yeni | szk? ", end="")
    print("\n" + "-" * 90)

    start = time.time()
    while time.time() - start < args.seconds:
        ok, fr = cap.read()
        if not ok:
            continue
        n += 1
        print(f"{n:>5} ", end="")
        for tag, old_m, new_m, thr, sz in models:
            oc = max_conf(old_m(fr, conf=0.05, imgsz=sz, verbose=False))
            nc = max_conf(new_m(fr, conf=0.05, imgsz=sz, verbose=False))
            verd = "Y" if nc >= thr else "-"
            print(f"| {oc:6.2f} | {nc:6.2f} |  {verd}  ", end="")
        print()
    cap.release()
    print(f"\n{n} kare toplandi. esikler: weapon {config.CONF} / cigarette {config.GREEN_CONF}, "
          f"imgsz: weapon {config.IMGSZ} / cigarette {config.GREEN_IMGSZ}")


if __name__ == "__main__":
    raise SystemExit(main())
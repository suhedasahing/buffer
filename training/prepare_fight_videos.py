"""Kavga video klasorlerini kare serilerine cevirir ve label manifest yazar.

Girdi yapilari (ikisinden biri):
  A) --root "fight"/"nofight" alt klasorleri ile  (0=kavga yok, 1=kavga)
  B) --csv <dosya> ; sutunlar: clip_path,label   (label 0/1)

Cikti:
  <out>/frames/<label>/<clip>_<idx>.jpg
  <out>/clips.csv   (clip,label,fps,w,h,frames -> cekilen kare sayisi)

Her klipte max --max-frames esit aralikli kare orneklenir (uzun CCTV klipleri icin).
Kalite kontrolu: acilamayan video atlanir, rapora yazilir.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2


def sample(video: Path, label: int, out_dir: Path, max_frames: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0:
        cap.release()
        return None
    stride = max(1, total // max_frames)
    idxs = list(range(0, total, stride))[:max_frames]
    n = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in idxs:
            dst = out_dir / f"{video.stem}_{i:05d}.jpg"
            cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            n += 1
        i += 1
    cap.release()
    return {"clip": video.stem, "label": label, "fps": fps, "w": w, "h": h, "frames": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="fight/nofight alt klasorlu kok dizin")
    ap.add_argument("--csv", help="clip_path,label iceren csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=40)
    args = ap.parse_args()

    videos = []
    if args.root:
        root = Path(args.root)
        for label_dir in ("fight", "nofight", "violent", "non-violent"):
            d = root / label_dir
            if d.is_dir():
                label = 0 if label_dir in ("nofight", "non-violent") else 1
                for v in sorted(d.glob("*")):
                    if v.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                        videos.append((v, label))
    elif args.csv:
        with open(args.csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                videos.append((Path(row["clip_path"]), int(row["label"])))
    else:
        ap.error("--root veya --csv verilmeli")

    out = Path(args.out)
    for label in ("0", "1"):
        (out / "frames" / label).mkdir(parents=True, exist_ok=True)

    rows, skipped = [], []
    for video, label in videos:
        info = sample(video, label, out / "frames" / str(label), args.max_frames)
        if info is None:
            skipped.append(str(video))
        else:
            rows.append(info)

    with open(out / "clips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "label", "fps", "w", "h", "frames"])
        w.writeheader()
        w.writerows(rows)

    print(f"islenen={len(rows)} atlanan={len(skipped)}")
    if skipped:
        print("ATLANAN:", skipped[:20])
    labs = {}
    for r in rows:
        labs[r["label"]] = labs.get(r["label"], 0) + 1
    print("label dagilimi:", labs)


if __name__ == "__main__":
    sys.exit(main())
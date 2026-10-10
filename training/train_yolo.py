"""YOLOv8/11 egitim CLI - Colab T4 icin hazir.

Kullanim (Colab'de):
    !python training/train_yolo.py --data training/data/weapon_v2/data.yaml --model yolov8n --epochs 80 --imgsz 640 --batch auto --device 0 --name weapon_v2

GPUsuz calismada --device cpu. Model ciktisi: runs/<name>/weights/best.pt
"""
import argparse
import gc
import json
import sys

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", default=-1)
    ap.add_argument("--device", default=0)
    ap.add_argument("--name", default="run")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="Veri setinin egitim icin kullanilacak orani (0-1). Hackathon hizi icin 0.3-0.5.")
    args = ap.parse_args()

    raw_batch = str(args.batch).strip().lower()
    batch = -1 if raw_batch in ("auto", "-1") else int(float(raw_batch))

    m = YOLO(args.model)
    results = m.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        fraction=args.fraction,
        plots=True,
    )

    best_path = f"{args.project}/{args.name}/weights/best.pt"
    summary = {
        "best": best_path,
        "metrics": {k: float(v) for k, v in results.results_dict.items()},
    }
    with open(f"{args.project}/{args.name}/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    gc.collect()


if __name__ == "__main__":
    sys.exit(main())
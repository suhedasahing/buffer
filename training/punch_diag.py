"""Tek seferlik kavga teşhis aracı: kavga neden tetiklenmiyor?

fight.py'nin score_pairs mantığını kare kare loglar. İki kişi yumruklaşırken
çalıştırın; çıktıda ya 'kisiler' düşük (kişi algılanmıyor), ya hız düşük
(yumruk vuruş hızı ölçülmüyor), ya da geometri reddediliyor (MESAFE/GEOM)
görürsünüz. Bu logların 8-10 satırını yapıştırın; düzeltme o kanala göre yapılır.

Kullanım: .venv\\Scripts\\python.exe training\\punch_diag.py --seconds 8
"""
import argparse
import time
import numpy as np
from pathlib import Path
import sys
import cv2
import itertools
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend import config  # noqa: E402

LIMBS = [9, 10, 13, 14]  # uretimle ayni: sadece kollar (bilek+dirsek)
NAMES = {9: "bilekL", 10: "bilekR", 13: "dirsekL", 14: "dirsekR"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--seconds", type=int, default=8)
    args = ap.parse_args()

    model = YOLO(config.POSE_MODEL)
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev = {}
    t0 = time.time()
    frame_i = 0
    while time.time() - t0 < args.seconds:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        frame_i += 1
        res = model.track(frame, persist=True, conf=config.POSE_CONF,
                          imgsz=config.FIGHT_IMGSZ, verbose=False)[0]
        now = time.time()
        if res.boxes is None or res.boxes.id is None or res.keypoints is None:
            print(f"[{frame_i}] KISI_YOK (tracker bos)")
            prev = {}
            continue
        ids = res.boxes.id.int().tolist()
        boxes = res.boxes.xyxy.cpu().numpy()
        kxy = res.keypoints.xy.cpu().numpy()
        kcf = res.keypoints.conf.cpu().numpy()

        people, new_prev = [], {}
        for i, pid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i]
            h = max(float(y2 - y1), 1.0)
            pts, ok_k = kxy[i][LIMBS], kcf[i][LIMBS] > config.POSE_KEY_CONF
            speeds = np.zeros(len(LIMBS))
            prev_pts = None
            if pid in prev:
                t0p, p0, ok0 = prev[pid]
                prev_pts = p0
                dt = max(now - t0p, 0.03)
                speeds = np.where(ok_k & ok0, np.linalg.norm(pts - p0, axis=1) / h / dt, 0.0)
            new_prev[pid] = (now, pts, ok_k)
            people.append({"id": pid, "box": (x1, y1, x2, y2), "h": h,
                           "pts": pts, "ok": ok_k, "speeds": speeds,
                           "center": np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])})
        prev = new_prev

        info = []
        for p in people:
            parts = []
            for j, idx in enumerate(LIMBS):
                name = NAMES[idx]
                if p["ok"][j]:
                    parts.append(f"{name}={p['speeds'][j]:.2f}")
                else:
                    parts.append(f"{name}=X")
            info.append(f"#{p['id']}[h={int(p['h'])}]: " + " ".join(parts))
        print(f"[{frame_i}] kisiler={len(people)}")
        for line in info:
            print("   ", line)

        if len(people) >= 2:
            for a, b in itertools.combinations(people, 2):
                ca, cb = a["center"], b["center"]
                d = float(np.linalg.norm(cb - ca))
                lim = config.FIGHT_NEAR * max(a["h"], b["h"])
                if d > lim:
                    print(f"   2 kisi AMA MESAFE_TOPLAM: d/h={d/max(a['h'],b['h']):.2f} "
                          f"esik={config.FIGHT_NEAR} (birbirinizden uzak duruyorsaniz normldur)")
                    continue
                for atk, vic, side in ((a, b, cb - ca), (b, a, ca - cb)):
                    bx1, by1, bx2, by2 = vic["box"]
                    m = config.FIGHT_MARGIN * d + 0.15 * vic["h"]
                    hit_cand = []
                    for j, (sp, (px, py), okp) in enumerate(zip(atk["speeds"], atk["pts"], atk["ok"])):
                        if not okp or sp < config.FIGHT_SPEED:
                            continue
                        in_box = bx1 - m <= px <= bx2 + m and by1 - m <= py <= by2 + m
                        forward = np.dot(np.array([px, py]) - atk["center"], side) >= 0
                        hit_cand.append(f"{NAMES[LIMBS[j]]}:sp={sp:.2f} kutuda={in_box} ileri={forward}")
                    if hit_cand:
                        for hc in hit_cand:
                            print(f"   ATK#{atk['id']}->VIC#{vic['id']}: {hc}")
                    else:
                        print(f"   ATK#{atk['id']}->VIC#{vic['id']}: hiz yetersiz (max "
                              f"{max(atk['speeds']):.2f} < {config.FIGHT_SPEED})")
        print()
    cap.release()


if __name__ == "__main__":
    main()
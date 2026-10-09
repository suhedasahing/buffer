"""Poz tabanlı kavga/vurma sezgisi: yakın iki kişi + biri diğerine doğru çok hızlı el/ayak hareketi."""
import time
import numpy as np
from ultralytics import YOLO
from . import config

LIMBS = [9, 10, 15, 16]  # COCO: sol/sağ bilek, sol/sağ ayak bileği


def _center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def score_pairs(people):
    """people: [{box, h, pts[4,2], speeds[4]}] -> (hız, birleşik_kutu) | None"""
    best, out = 0.0, None
    for a in people:
        for b in people:
            if a is b:
                continue
            if np.linalg.norm(_center(a["box"]) - _center(b["box"])) > config.FIGHT_NEAR * max(a["h"], b["h"]):
                continue
            bx1, by1, bx2, by2 = b["box"]
            m = 0.15 * b["h"]
            for sp, (px, py) in zip(a["speeds"], a["pts"]):
                if sp >= config.FIGHT_SPEED and sp > best and bx1 - m <= px <= bx2 + m and by1 - m <= py <= by2 + m:
                    best = sp
                    out = (int(min(a["box"][0], bx1)), int(min(a["box"][1], by1)),
                           int(max(a["box"][2], bx2)), int(max(a["box"][3], by2)))
    return (best, out) if out else None


class FightDetector:
    def __init__(self):
        self.model = YOLO(config.POSE_MODEL)
        self.prev = {}  # track_id -> (zaman, noktalar, geçerlilik)

    def infer(self, frame):
        res = self.model.track(frame, persist=True, conf=0.35, imgsz=config.FIGHT_IMGSZ, verbose=False)[0]
        if res.boxes is None or res.boxes.id is None or res.keypoints is None:
            self.prev = {}
            return []
        now = time.time()
        ids = res.boxes.id.int().tolist()
        boxes = res.boxes.xyxy.cpu().numpy()
        kxy = res.keypoints.xy.cpu().numpy()
        kcf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])
        people, new_prev = [], {}
        for i, pid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i]
            h = max(float(y2 - y1), 1.0)
            pts, ok = kxy[i][LIMBS], kcf[i][LIMBS] > 0.4
            speeds = np.zeros(len(LIMBS))
            if pid in self.prev:
                t0, p0, ok0 = self.prev[pid]
                dt = max(now - t0, 0.03)
                speeds = np.where(ok & ok0, np.linalg.norm(pts - p0, axis=1) / h / dt, 0.0)
            new_prev[pid] = (now, pts, ok)
            people.append({"box": (x1, y1, x2, y2), "h": h, "pts": pts, "speeds": speeds})
        self.prev = new_prev
        if len(people) < 2:
            return []
        r = score_pairs(people)
        if not r:
            return []
        sp, box = r
        return [{"label": "fight", "conf": min(0.99, 0.5 + 0.05 * sp), "level": "warning", "box": box}]

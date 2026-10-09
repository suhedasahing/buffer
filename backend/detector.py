import cv2
from ultralytics import YOLO
from . import config

COLORS = {"warning": (0, 190, 255), "critical": (60, 60, 255)}  # BGR


def level_of(label: str):
    l = label.lower()
    if l in config.CRITICAL_CLASSES:
        return "critical"
    if l in config.WARNING_CLASSES:
        return "warning"
    return None


class ThreatDetector:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)
        print(f"[detector] model={config.MODEL_PATH} classes={list(self.model.names.values())[:20]}")

    def infer(self, frame):
        """Dönen: [{label, conf, level, box:(x1,y1,x2,y2)}] sadece tehdit sınıfları"""
        res = self.model(frame, conf=config.CONF, imgsz=config.IMGSZ, verbose=False)[0]
        out = []
        for b in res.boxes:
            label = self.model.names[int(b.cls)]
            lvl = level_of(label)
            if not lvl:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            out.append({"label": label, "conf": float(b.conf), "level": lvl, "box": (x1, y1, x2, y2)})
        return out

    @staticmethod
    def draw(frame, dets, cam_name, state):
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            c = COLORS[d["level"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f'{d["label"]} {d["conf"]:.2f}', (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        bar = {"normal": (60, 160, 60), "warning": COLORS["warning"], "critical": COLORS["critical"]}[state]
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), bar, -1)
        cv2.putText(frame, f"{cam_name} | {state.upper()}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

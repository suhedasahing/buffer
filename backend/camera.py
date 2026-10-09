import os, time, threading
from collections import deque
import cv2
from . import config
from .detector import ThreatDetector


class CameraWorker(threading.Thread):
    """Kamera -> silah/bıçak (YOLO) + kavga (poz) -> zaman filtresi -> olay."""

    def __init__(self, cam_id, cfg, detector: ThreatDetector, on_event, fight_model=None):
        super().__init__(daemon=True)
        self.cam_id, self.name_, self.source = cam_id, cfg["name"], cfg["source"]
        self.detector, self.on_event = detector, on_event
        self.fight_model = fight_model  # eğitilmiş kavga/tırmanma modeli (opsiyonel)
        self.fight = None
        if config.ENABLE_FIGHT:
            from .fight import FightDetector
            self.fight = FightDetector()  # her kameranın kendi takip durumu olmalı
        self.history = deque(maxlen=config.WINDOW)
        self.last_alert = {"warning": 0.0, "critical": 0.0}
        self.lock = threading.Lock()
        self.jpeg = None
        self.last_frame = None
        self.state = "normal"
        self.fps = 0.0
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        t_prev = time.time()
        while self.running:
            ok, frame = cap.read()
            if not ok:
                if isinstance(self.source, str) and os.path.isfile(self.source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    time.sleep(1.0); cap.release(); cap = cv2.VideoCapture(self.source)
                continue

            dets = self.detector.infer(frame)
            if self.fight_model:
                dets += self.fight_model.infer(frame)
            if self.fight:
                dets += self.fight.infer(frame)
            top = max(dets, key=lambda d: (d["level"] == "critical", d["conf"])) if dets else None
            self.history.append((top["level"], top["label"], top["conf"]) if top else None)

            state, label, conf = self._decide()
            self.state = state
            frame = self.detector.draw(frame, dets, self.name_, state)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self.lock:
                self.last_frame = frame
                if ok:
                    self.jpeg = buf.tobytes()

            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1 / max(now - t_prev, 1e-3))
            t_prev = now

            if state != "normal" and now - self.last_alert[state] > config.COOLDOWN_SEC:
                self.emit(state, label, conf)
        cap.release()

    def _decide(self):
        hits = [h for h in self.history if h]
        crit = [h for h in hits if h[0] == "critical"]
        warn = [h for h in hits if h[0] == "warning"]
        fight = [h for h in warn if h[1] == "fight"]
        if len(crit) >= config.MIN_HITS:
            b = max(crit, key=lambda h: h[2]); return "critical", b[1], b[2]
        if len(warn) >= config.MIN_HITS or len(fight) >= config.FIGHT_MIN_HITS:
            b = max(warn, key=lambda h: h[2]); return "warning", b[1], b[2]
        return "normal", None, 0.0

    def emit(self, level, label, conf):
        self.last_alert[level] = time.time()
        ts = int(time.time() * 1000)
        snap = f"{self.cam_id}_{ts}.jpg"
        with self.lock:
            if self.last_frame is not None:
                cv2.imwrite(str(config.SNAP_DIR / snap), self.last_frame)
        self.on_event({
            "camera": self.cam_id, "camera_name": self.name_, "level": level,
            "label": label or "simulated", "conf": round(float(conf), 2),
            "ts": ts, "snapshot": f"/snapshots/{snap}",
        })

    def get_jpeg(self):
        with self.lock:
            return self.jpeg

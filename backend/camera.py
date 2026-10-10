import os, time, threading
from collections import deque
import cv2
from . import config
from .detector import ThreatDetector

# Alarm önceliği: kırmızı > sarı > yeşil
RANK = {"green": 0, "warning": 1, "critical": 2}


def _open_cap(source):
    """Kamerayı açar ve mümkünse geniş açı/zoom'suz mod ister (bazı webcam'ler firmware'de yakınlaştırır)."""
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAP_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAP_HEIGHT)
            cap.set(cv2.CAP_PROP_ZOOM, 0)  # destekleyen kameralarda zoom'u sıfırlar
        except Exception:
            pass
    return cap


class CameraWorker(threading.Thread):
    """Kamera akışı ve model çıkarımı AYRI iş parçacıklarında çalışır.

    Böylece ağır YOLO çıkarımı video akışını (FPS) bloke etmez: akış kameranın
    hızında akıcı kalır, tespit sonuçları arkada güncellenir.
    """

    def __init__(self, cam_id, cfg, detector: ThreatDetector, on_event, fight_model=None):
        super().__init__(daemon=True)
        self.cam_id, self.name_, self.source = cam_id, cfg["name"], cfg["source"]
        self.detector, self.on_event = detector, on_event
        self.fight_model = fight_model  # eğitilmiş kavga/tırmanma modeli (opsiyonel)
        self.fight = None
        if config.ENABLE_FIGHT:
            from .fight import FightDetector
            self.fight = FightDetector()  # her kameranın kendi takip durumu olmalı
        # 🟢 YEŞİL ALARM: sigara modeli varsa yükle (yoksa None -> yeşil alarm pasif)
        self.green = None
        if config.CIGARETTE_MODEL_PATH:
            from .detector import GreenDetector
            from . import active_learning
            self.green = GreenDetector(config.CIGARETTE_MODEL_PATH)
            active_learning.register("cigarette", self.green.model_names)
        self.history = deque(maxlen=config.WINDOW)
        self.green_history = deque(maxlen=config.GREEN_WINDOW)  # 🟢 sigaraya özel geniş pencere
        self.last_state = ("normal", None, 0.0, None, None, None)  # 🧷 son karar (hold/stabilite için)
        self.last_state_ts = 0.0
        self.last_alert = {"green": 0.0, "warning": 0.0, "critical": 0.0}
        self.lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.latest_frame = None   # akış -> çıkarım (paylaşılan son kare)
        self.jpeg = None
        self.last_frame = None
        self.dets = []             # çıkarım -> akış (paylaşılan son tespitler)
        self.state = "normal"
        self.fps = 0.0
        self.running = True
        self.infer_pass = 0        # çıkarım turu sayacı (green/poz modelini seyreltmek için)

    def run(self):
        """Video akışı döngüsü (hızlı): oku -> çiz -> JPEG kodla. Model çalıştırmaz."""
        threading.Thread(target=self._infer_loop, daemon=True).start()
        cap = _open_cap(self.source)
        t_prev = time.time()
        while self.running:
            ok, frame = cap.read()
            if not ok:
                if isinstance(self.source, str) and os.path.isfile(self.source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    time.sleep(0.5); cap.release(); cap = _open_cap(self.source)
                continue
            with self.frame_lock:
                self.latest_frame = frame
            out = self.detector.draw(frame, self.dets, self.name_, self.state)
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self.lock:
                self.last_frame = out
                if ok:
                    self.jpeg = buf.tobytes()
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1 / max(now - t_prev, 1e-3))
            t_prev = now
        cap.release()

    def _infer_loop(self):
        """Model çıkarımı döngüsü (yavaş): son kareyi al -> çalıştır -> karar/olay üret."""
        while self.running:
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is None:
                time.sleep(0.02); continue
            t0 = time.time()
            self.infer_pass += 1
            # 🥊 SADECE-KAVGA modu: silah çıkarımı yapılmaz; sadece kavga (pose) çalışır.
            dets = [] if config.ONLY_FIGHT else list(self.detector.infer(frame))  # 🔴 silah (her turda)
            if self.green and self.infer_pass % config.GREEN_EVERY == 0:
                dets += self.green.infer(frame)        # 🟢 sigara
            if self.infer_pass % config.FIGHT_EVERY == 0:
                if self.fight_model:
                    dets += self.fight_model.infer(frame)  # 🟡 fight.pt
                if self.fight:
                    dets += self.fight.infer(frame)        # 🟡 kavga + 🔴 ROI/tırmanma
            with self.lock:
                self.dets = dets

            # Bu turda HER seviyenin en iyi tespitini sakla: tek "top" yüzünden
            # (ör. silah gürültüsü) sigara/kavga tespitlerinin gölgelenmesini önler.
            best = {}
            for d in dets:
                lvl = d["level"]
                if lvl not in best or d["conf"] > best[lvl]["conf"]:
                    best[lvl] = d
            self.history.append(best or None)
            self.green_history.append(best.get("green") if best else None)
            state, label, conf, source, box, cls = self._decide()
            self.state = state

            now = time.time()
            if state != "normal" and now - self.last_alert[state] > config.COOLDOWN_SEC:
                nb = None
                if box is not None:
                    fh, fw = frame.shape[:2]
                    x1, y1, x2, y2 = box
                    nb = ((x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh, (x2 - x1) / fw, (y2 - y1) / fh)
                self.emit(state, label, conf, source, nb, cls)

            # Çıkarımı sınırla: CPU'yu video akışına bırak.
            time.sleep(max(0.0, config.INFER_INTERVAL - (time.time() - t0)))

    def _decide(self):
        """3 kademeli karar: 🔴 kritik > 🟡 uyarı/kavga > 🟢 yeşil (sigara)."""
        cur = self.history[-1] if self.history else None
        # 1) ⚡ ANINDA TETİKLEME: çok yüksek güvenli tespit, vuruş birikmesini beklemeden alarm (hız).
        for lvl, d in (("critical", cur.get("critical") if cur else None),
                       ("warning",  cur.get("warning")  if cur else None),
                       ("green",    self.green_history[-1] if self.green_history else None)):
            if d and d["conf"] >= config.IMMEDIATE_CONF:
                self._store(lvl, d)
                return lvl, d["label"], d["conf"], d.get("source"), d.get("box"), d.get("cls")

        # 2) BİRİKİMLİ ONAY (yanlış alarm filtresi)
        crit = [h["critical"] for h in self.history if h and "critical" in h]
        warn = [h["warning"] for h in self.history if h and "warning" in h]
        green = [g for g in self.green_history if g]  # 🟢 kendi geniş penceresinde sayılır
        fight = [w for w in warn if w["label"] == "fight"]
        if len(crit) >= config.CRIT_MIN_HITS:
            b = max(crit, key=lambda d: d["conf"]); res = ("critical", b["label"], b["conf"], b.get("source"), b.get("box"), b.get("cls"))
        elif len(warn) >= config.MIN_HITS or len(fight) >= config.FIGHT_MIN_HITS:
            b = max(warn, key=lambda d: d["conf"]); res = ("warning", b["label"], b["conf"], b.get("source"), b.get("box"), b.get("cls"))
        elif len(green) >= config.GREEN_MIN_HITS:
            b = max(green, key=lambda d: d["conf"]); res = ("green", b["label"], b["conf"], b.get("source"), b.get("box"), b.get("cls"))
        else:
            res = ("normal", None, 0.0, None, None, None)

        # 3) 🧷 STABİLİTE: tespit kısa süre kesildiyse durumu hemen düşürme; STATE_HOLD_SEC boyunca koru.
        now = time.time()
        if res[0] == "normal" and self.last_state[0] != "normal" and now - self.last_state_ts < config.STATE_HOLD_SEC:
            return self.last_state
        self.last_state, self.last_state_ts = res, now
        return res

    def _store(self, lvl, d):
        self.last_state = (lvl, d["label"], d["conf"], d.get("source"), d.get("box"), d.get("cls"))
        self.last_state_ts = time.time()

    def emit(self, level, label, conf, source=None, box=None, cls=None):
        self.last_alert[level] = time.time()
        print(f"[alarm] {self.cam_id} level={level} label={label} conf={conf} source={source}")
        ts = int(time.time() * 1000)
        snap = f"{self.cam_id}_{ts}.jpg"
        with self.lock:
            if self.last_frame is not None:
                cv2.imwrite(str(config.SNAP_DIR / snap), self.last_frame)
        self.on_event({
            "camera": self.cam_id, "camera_name": self.name_, "level": level,
            "label": label or "simulated", "conf": round(float(conf), 2),
            "source": source or "", "box": box, "cls": cls,
            "ts": ts, "snapshot": f"/snapshots/{snap}",
        })

    def get_jpeg(self):
        with self.lock:
            return self.jpeg

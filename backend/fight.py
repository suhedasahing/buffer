"""Poz tabanlı kavga/vurma sezgisi + 🔴 intihar/camdan atlama (ROI) tespiti.

Tek bir poz (pose) modeli hem kavgayı hem de "sanal pencere bölgesi"ni besler;
böylece performans için aynı döngüde ekstra model çalıştırılmaz.
"""
import time
import itertools
import numpy as np
from ultralytics import YOLO
from . import config

LIMBS = [9, 10, 13, 14]  # COCO: sol/sağ bilek + sol/sağ dirsek (SADECE KOLLAR).
# Ayak/bileği skorlamadan çıkardık: yürüyüşte ayak hızı kolayca eşiği geçip sahte alarm üretiyordu
# (ölçüm: 1.9-2.0 ile geri adım). Yumruk için kol noktaları yeterli ve çok daha temiz.


def _center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def score_pairs(people):
    """people: [{box, h, pts[4,2], speeds[4]}] -> (hız, birleşik_kutu) | None

    Vuruş, saldıranın uzvunun KARŞI kişinin kutusunun İÇİNE girmesini şart koşmaz;
    kutu, iki kişi arasındaki mesafenin FIGHT_MARGIN katı kadar genişletilir. Böylece
    kol boşluğun ortasını (midline) geçtiyse — yani karşıya doğru savrulan yumruk —
    uzaktaki/aralıklı kavgalarda bile sayılır. Uzvun "dışa/arkaya" değil, hedefe doğru
    tarafta olması şartı sahte pozitifleri bastırır.
    """
    best, out = 0.0, None
    for a, b in itertools.combinations(people, 2):
        ca, cb = _center(a["box"]), _center(b["box"])
        toward = cb - ca  # a -> b yönü
        if np.linalg.norm(toward) > config.FIGHT_NEAR * max(a["h"], b["h"]):
            continue
        # Her iki yönü de dene: (a->b) ve (b->a)
        for atk, vic, c_vic, side in ((a, b, cb, toward), (b, a, ca, -toward)):
            bx1, by1, bx2, by2 = vic["box"]
            d = float(np.linalg.norm(side))
            m = config.FIGHT_MARGIN * d + 0.15 * vic["h"]
            for sp, (px, py) in zip(atk["speeds"], atk["pts"]):
                if sp < config.FIGHT_SPEED or sp <= best:
                    continue
                if not (bx1 - m <= px <= bx2 + m and by1 - m <= py <= by2 + m):
                    continue
                # Uzuv, saldıranın ARKASINDA (hedefin ters tarafında) olmasın.
                if np.dot(np.array([px, py]) - ca, side) < 0:
                    continue
                best = sp
                out = (int(min(a["box"][0], bx1)), int(min(a["box"][1], by1)),
                       int(max(a["box"][2], bx2)), int(max(a["box"][3], by2)))
    return (best, out) if out else None


class FightDetector:
    def __init__(self):
        self.model = YOLO(config.POSE_MODEL)
        # pid -> (zaman, noktalar, geçerlilik, merkez)  (merkez, ROI/atlama tespiti için)
        self.prev = {}
        # 🟡 ŞÜPHELİ BEKLEŞ (loitering): pid -> (ilk görülme zamanı, birikmiş hareketsiz süre)
        self.entry_times = {}
        self.still_time = {}
        # 🔴 ROI/atlama doğrulama sayaçları ve tetikleme kilidi (latch)
        # Kilit sayesinde bölgede SÜREKLİ duran kişi tekrar tekrar alarm üretmez;
        # yeniden tetiklenmesi için kişinin bölgeden çıkması gerekir.
        self.roi_hits = 0
        self.roi_armed = True
        self.drop_hits = 0
        self.drop_armed = True
        # 🔴 Tırmanma (yukarı hareket) doğrulama sayacı ve kilidi
        self.rise_hits = 0
        self.rise_armed = True

    def infer(self, frame):
        res = self.model.track(frame, persist=True, conf=config.POSE_CONF, imgsz=config.FIGHT_IMGSZ, verbose=False)[0]
        if res.boxes is None or res.boxes.id is None or res.keypoints is None:
            self.prev = {}
            self.entry_times = {}
            self.still_time = {}
            self.roi_hits = self.drop_hits = self.rise_hits = 0
            self.roi_armed = self.drop_armed = self.rise_armed = True
            return []
        now = time.time()
        ids = res.boxes.id.int().tolist()
        boxes = res.boxes.xyxy.cpu().numpy()
        kxy = res.keypoints.xy.cpu().numpy()
        kcf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])
        # Sahnede artık olmayan kişilerin loiter takibini sıfırla
        active = set(ids)
        self.entry_times = {p: t for p, t in self.entry_times.items() if p in active}
        self.still_time = {p: v for p, v in self.still_time.items() if p in active}
        people, new_prev = [], {}
        for i, pid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i]
            h = max(float(y2 - y1), 1.0)
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])  # kişi merkez koordinatı
            pts, ok = kxy[i][LIMBS], kcf[i][LIMBS] > config.POSE_KEY_CONF
            speeds = np.zeros(len(LIMBS))
            prev_center = None
            dt = 0.0
            if pid in self.prev:
                t0, p0, ok0, c0 = self.prev[pid]
                prev_center = c0
                dt = max(now - t0, 0.03)
                speeds = np.where(ok & ok0, np.linalg.norm(pts - p0, axis=1) / h / dt, 0.0)
            new_prev[pid] = (now, pts, ok, center)
            people.append({"box": (x1, y1, x2, y2), "h": h, "pts": pts, "speeds": speeds,
                           "center": center, "prev_center": prev_center})

            # --- 🟡 Şüpheli bekleyiş (loitering): sahnede kalma süresi + hareketsizlik takibi ---
            if config.ENABLE_LOITERING:
                if pid not in self.entry_times:
                    self.entry_times[pid] = now
                    self.still_time[pid] = 0.0
                if prev_center is not None and dt > 0:
                    disp = float(np.linalg.norm(center - prev_center)) / h
                    max_sp = float(np.max(speeds)) if len(speeds) else 0.0
                    still = max_sp < config.LOITERING_STILL_SPEED and disp < config.LOITERING_STILL_MOVE
                    self.still_time[pid] = self.still_time[pid] + dt if still else 0.0
        self.prev = new_prev

        events = []

        # --- 🟡 Şüpheli bekleyiş (loitering): yeterince kalıp hareketsiz duran tek kişi ---
        if config.ENABLE_LOITERING:
            for pid, p in zip(ids, people):
                dwell = now - self.entry_times.get(pid, now)
                if (dwell >= config.LOITERING_TIME and
                        dwell < config.LOITERING_TIME + config.LOITERING_WINDOW and
                        self.still_time.get(pid, 0.0) >= config.LOITERING_TIME):
                    events.append({"label": "loitering", "conf": 0.90, "level": "warning",
                                   "box": (int(p["box"][0]), int(p["box"][1]),
                                           int(p["box"][2]), int(p["box"][3])), "source": "pose"})

        # --- 🟡 Kavga tespiti (mevcut mantık) ---
        if config.DEBUG_FIGHT:
            sp = max((max(p["speeds"]) for p in people), default=0.0)
            print(f"[fight] kisiler={len(people)} maks_uzuv_hizi={sp:.2f} esik={config.FIGHT_SPEED} "
                  f"({'2+ kisi gerekli' if len(people) < 2 else 'kontrol ediliyor'})")
        if len(people) >= 2:
            r = score_pairs(people)
            if config.DEBUG_FIGHT:
                print(f"[fight] kavga_skoru={r[0] if r else 0.0:.2f} esik={config.FIGHT_SPEED} "
                      f"({'TETIKLENDI' if r else 'vurus yok'})")
            if r:
                sp, box = r
                events.append({"label": "fight", "conf": min(0.99, 0.5 + 0.05 * sp),
                               "level": "warning", "box": box, "source": "pose"})

        # --- 🔴 İntihar / camdan atlama tespiti ---
        danger = self._window_danger(people, frame.shape)
        if danger:
            events.append(danger)
        return events

    def _window_danger(self, people, shape):
        """Sanal pencere ROI'sine giren veya y ekseninde ani düşüş yaşayan kişiyi bulur.

        ROI_ZONE normalize (0-1) koordinatlardan gerçek piksel koordinatlarına burada çevrilir.
        Tek karelik sıçramalarda yanlış alarmı önlemek için ardışık kare sayacı kullanılır.
        """
        if not config.ROI_ENABLED or not people:
            self.roi_hits = self.drop_hits = self.rise_hits = 0
            self.roi_armed = self.drop_armed = self.rise_armed = True
            return None

        H, W = shape[:2]
        x1n, y1n, x2n, y2n = config.ROI_ZONE
        # ROI_ZONE'un piksel karşılığı (tehlikeli bölge kutusu)
        rx1, ry1, rx2, ry2 = x1n * W, y1n * H, x2n * W, y2n * H

        any_inside = any_drop = any_rise = False
        inside_person = drop_person = rise_person = None
        for p in people:
            cx, cy = p["center"]
            # 1) Merkez tehlikeli bölgenin içinde mi?
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                any_inside = True
                inside_person = p
            # 2) Ani y düşüşü (camdan atlama/düşme): merkezin dikey hızı vücut boyuna oranı
            if p["prev_center"] is not None:
                drop = (cy - p["prev_center"][1]) / p["h"]  # y aşağı arttığı için + = düşüş
                if drop >= config.DROP_RATIO:
                    any_drop = True
                    drop_person = p
                # 3) Yukarı hareket (tırmanma/duvara-pencereye çıkma): negatif = yukarı.
                #    Kişi merkezi karenin üst kısmındayken (CLIMB_TOP) anlamlıdır.
                rise = (p["prev_center"][1] - cy) / p["h"]  # + = yukarı
                if rise >= config.RISE_RATIO and cy <= config.CLIMB_TOP * H:
                    any_rise = True
                    rise_person = p

        # --- Ardışık kare sayaçları + latch (kilit) ---
        # ROI: kişi bölgeye girip ROI_MIN_HITS kare kalırsa BİR kez tetiklenir; bölgeden
        # çıkana kadar (boş kalana kadar) tekrar tetiklenmez.
        if any_inside:
            self.roi_hits += 1
        else:
            self.roi_hits = 0
            self.roi_armed = True
        roi_trig = self.roi_armed and self.roi_hits >= config.ROI_MIN_HITS
        if roi_trig:
            self.roi_armed = False

        # Atlama/düşüş: aynı latch mantığı.
        if any_drop:
            self.drop_hits += 1
        else:
            self.drop_hits = 0
            self.drop_armed = True
        drop_trig = self.drop_armed and self.drop_hits >= config.DROP_MIN_HITS
        if drop_trig:
            self.drop_armed = False

        # Tırmanma/yukarı hareket: aynı latch mantığı.
        if any_rise:
            self.rise_hits += 1
        else:
            self.rise_hits = 0
            self.rise_armed = True
        climb_trig = self.rise_armed and self.rise_hits >= config.RISE_MIN_HITS
        if climb_trig:
            self.rise_armed = False

        if not (roi_trig or drop_trig or climb_trig):
            return None

        if roi_trig:
            p = inside_person
        elif drop_trig:
            p = drop_person
        else:
            p = rise_person
        if p is None:
            p = people[0]
        bx1, by1, bx2, by2 = (int(v) for v in p["box"])
        label = "climb" if climb_trig else "suicide"
        return {"label": label, "conf": 0.90, "level": "critical", "box": (bx1, by1, bx2, by2), "source": "pose"}

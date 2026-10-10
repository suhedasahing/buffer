import cv2
from ultralytics import YOLO
from . import config

COLORS = {"green": (80, 200, 90), "warning": (0, 190, 255), "critical": (60, 60, 255),
          "info": (255, 0, 255)}  # BGR (info = mor, şüpheli bekleyiş çerçevesi)


def level_of(label: str):
    l = label.lower()
    if l in config.CRITICAL_CLASSES:
        return "critical"
    if l in config.WARNING_CLASSES:
        return "warning"
    # Yeşil sınıflar farklı modellerde farklı isimlerle gelebilir (ör. "Smoking - v5 Detectron2").
    # Bu yüzden yeşil için hem tam eşleşme hem de anahtar-kelime (substring) kontrolü yapılır.
    if l in config.GREEN_CLASSES or any(k in l for k in config.GREEN_KEYWORDS):
        return "green"
    return None


class ThreatDetector:
    def __init__(self, model_path: str | None = None, conf: float | None = None, source: str = "weapon"):
        self.model = YOLO(model_path or config.MODEL_PATH)
        self.conf = config.CONF if conf is None else conf
        self.source = source  # hangi model üretti? (weapon / fight_model)
        print(f"[detector] source={source} model={model_path or config.MODEL_PATH} conf={self.conf} "
              f"iou={config.IOU} classes={list(self.model.names.values())[:20]}")

    def infer(self, frame):
        """Dönen: [{label, conf, level, box:(x1,y1,x2,y2), source}] sadece tehdit sınıfları"""
        # Ana silah modeli eşiği ARAYÜZDEN canlı ayarlanabilsin diye config.CONF okunur.
        conf = config.CONF if self.source == "weapon" else self.conf
        res = self.model(frame, conf=conf, iou=config.IOU, imgsz=config.IMGSZ,
                         agnostic_nms=config.AGNOSTIC_NMS, max_det=config.MAX_DET, verbose=False)[0]
        # DEBUG: eşiği ne olursa olsun HAM tespitleri (düşük güvenliler dahil) yazdır -> DEBUG_DET=1
        if config.DEBUG_DET:
            dbg = self.model(frame, conf=0.05, iou=config.IOU, imgsz=config.IMGSZ, verbose=False)[0]
            found = [(self.model.names[int(b.cls)], round(float(b.conf), 2)) for b in dbg.boxes]
            if found:
                print(f"[raw:{self.source}] conf>={conf} -> {found}")
        out = []
        for b in res.boxes:
            label = self.model.names[int(b.cls)]
            lvl = level_of(label)
            if not lvl:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            out.append({"label": label, "conf": float(b.conf), "level": lvl,
                        "box": (x1, y1, x2, y2), "source": self.source, "cls": int(b.cls)})
        return out

    @property
    def model_names(self):
        return [self.model.names[i] for i in sorted(self.model.names)]

    @staticmethod
    def draw(frame, dets, cam_name, state):
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            c = COLORS["info"] if d["label"] == "loitering" else COLORS[d["level"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f'{d["label"]} {d["conf"]:.2f}', (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        bar = {"normal": (60, 160, 60), "green": COLORS["green"],
               "warning": COLORS["warning"], "critical": COLORS["critical"]}[state]
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), bar, -1)
        cv2.putText(frame, f"{cam_name} | {state.upper()}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame


class GreenDetector:
    """🟢 Sigara/yeşil alarm modeli (opsiyonel). models/cigarette.pt varsa yüklenir.

    weapon.pt içinde sigara sınıfı olmadığı için yeşil alarm ayrı bir modelle çalışır.
    Model yoksa bu dedektör hiç oluşturulmaz; yeşil alarm yalnızca /api/simulate ile test edilebilir.
    """

    def __init__(self, path):
        self.model = YOLO(path)
        # Bazı sigara modelleri bozuk etiketli gelir (ör. gerçek sınıf adı "undefined").
        # Arayüzde düzgün görünmesi için bu tür adları "cigarette" olarak eşliyoruz (salt-okunur model.names'e dokunmadan).
        self.name_fix = {i: "cigarette" for i, n in self.model.names.items()
                         if str(n).strip().lower() in ("undefined", "", "none")}
        self._names = {i: self.name_fix.get(i, n) for i, n in self.model.names.items()}
        print(f"[green] model={path} classes={list(self._names.values())[:20]}")

    def infer(self, frame):
        # DEBUG: sigara modelinin HAM çıktılarını görmek için -> DEBUG_DET=1
        if config.DEBUG_DET:
            dbg = self.model(frame, conf=0.05, iou=config.IOU, imgsz=config.GREEN_IMGSZ, verbose=False)[0]
            found = [(self._names.get(int(b.cls), self.model.names[int(b.cls)]), round(float(b.conf), 2)) for b in dbg.boxes]
            if found:
                print(f"[raw:cigarette] -> {found}")
        res = self.model(frame, conf=config.GREEN_CONF, iou=config.IOU, imgsz=config.GREEN_IMGSZ,
                         agnostic_nms=True, max_det=config.MAX_DET, verbose=False)[0]
        out = []
        for b in res.boxes:
            label = self._names.get(int(b.cls), self.model.names[int(b.cls)])
            if level_of(label) != "green":
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            out.append({"label": label, "conf": float(b.conf), "level": "green",
                        "box": (x1, y1, x2, y2), "source": "cigarette", "cls": int(b.cls)})
        return out

    @property
    def model_names(self):
        return [self._names[i] for i in sorted(self._names)]

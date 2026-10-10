"""Kavga videosu kaydet + offline teşhis: kavga neden tetiklenmiyor?

Faz 1: kameradan --seconds saniye MPJPG/.avi kaydeder (ekranda geri sayım
basılır, o sırada kamera önünde 2 kişi yumruklaşın).

Faz 2: video offline açılır, fight.py'nin mantığı kare kare loglanır ve
toplamlarla birlikte --log dosyasına yazılır.

Çıktıyı analiz etmesi için buraya atın (log dosyası otomatik yazılır).

Kullanım: .venv\\Scripts\\python.exe training\\record_and_diag.py --seconds 10
"""
import argparse
import time
import numpy as np
import cv2
import itertools
from pathlib import Path
import sys
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend import config  # noqa: E402

LIMBS = [9, 10, 13, 14]
NAMES = {9: "bilekL", 10: "bilekR", 13: "dirsekL", 14: "dirsekR"}
TMP = Path("C:/Users/suhed/AppData/Local/Temp/opencode")


def record(cam_id, seconds, out_path):
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps < 1:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    print(f"Kamera {cam_id}: {w}x{h} @ {fps:.0f}fps  (writer acik={writer.isOpened()})")
    if not writer.isOpened():
        raise SystemExit("VideoWriter acilamadi - codec sorunu")

    print("3 saniye sonra KAYIT basliyor - kameraya gecin ve yumruklasin!")
    for i in (3, 2, 1):
        print(f"...{i}")
        time.sleep(1)
    print("KAYIT BASLADI — YUMPRUKLASIN!")
    t0 = time.time()
    n = 0
    n_fail = 0
    empty = 0
    stamps = []
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            empty += 1
            time.sleep(0.01)
            continue
        if not writer.write(frame):
            n_fail += 1
        stamps.append(time.time())
        n += 1
    writer.release()
    cap.release()
    size_mb = out_path.stat().st_size / (1024 * 1024) if out_path.exists() else 0
    print(f"Kayit bitti: {n} kare ({seconds}s), bos kare: {empty}, yazilamayan: {n_fail}")
    print(f"Video: {out_path} ({size_mb:.1f} MB)")
    if size_mb < 0.1:
        raise SystemExit("Video dosyasi neredeyse bos - kayit basarisiz")
    stamp_path = out_path.with_suffix(".stamps.txt")
    stamp_path.write_text("\n".join(f"{s:.4f}" for s in stamps), encoding="utf-8")
    if len(stamps) >= 2:
        print(f"Gerçek FPS: {len(stamps)/max(stamps[-1]-stamps[0], 0.001):.1f}")
    print(f"Zaman damgalari: {stamp_path}")
    return out_path, fps


def analyze(video_path, log_path, key_conf):
    model = YOLO(config.POSE_MODEL)
    cap = cv2.VideoCapture(str(video_path))
    stamp_path = Path(str(video_path).replace(".avi", ".stamps.txt"))
    stamps = []
    if stamp_path.exists():
        stamps = [float(x) for x in stamp_path.read_text(encoding="utf-8").split()]

    out_lines = []
    def out(s=""):
        out_lines.append(s)
        print(s)

    prev = {}
    frame_i = 0
    n_kisi1 = n_kisi2 = n_en_fazla = n_hiz = n_mesafe = n_hit = 0
    best_hit = 0.0
    prev_ts = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if stamps:
            ts = stamps[min(frame_i - 1, len(stamps) - 1)]
            dt = max(ts - prev_ts, 0.03) if prev_ts is not None else 0.033
            prev_ts = ts
        else:
            dt = 0.033
        res = model.track(frame, persist=True, conf=config.POSE_CONF,
                          imgsz=config.FIGHT_IMGSZ, verbose=False)[0]
        if res.boxes is None or res.boxes.id is None or res.keypoints is None:
            out(f"[{frame_i}] KISI_YOK")
            prev = {}
            continue
        ids = res.boxes.id.int().tolist()
        boxes = res.boxes.xyxy.cpu().numpy()
        kxy = res.keypoints.xy.cpu().numpy()
        kcf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])

        people, new_prev = [], {}
        for i, pid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i]
            h = max(float(y2 - y1), 1.0)
            pts, ok_k = kxy[i][LIMBS], kcf[i][LIMBS] > key_conf
            speeds = np.zeros(len(LIMBS))
            if pid in prev:
                p0, ok0 = prev[pid]
                speeds = np.where(ok_k & ok0, np.linalg.norm(pts - p0, axis=1) / h / dt, 0.0)
            new_prev[pid] = (pts, ok_k)
            people.append({"id": pid, "box": (x1, y1, x2, y2), "h": h,
                           "pts": pts, "ok": ok_k, "speeds": speeds,
                           "center": np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])})
        prev = new_prev

        n_en_fazla = max(n_en_fazla, len(people))
        if len(people) == 1:
            n_kisi1 += 1
        if len(people) >= 2:
            n_kisi2 += 1

        info = []
        for p in people:
            parts = []
            for j, idx in enumerate(LIMBS):
                if p["ok"][j]:
                    parts.append(f"{NAMES[idx]}={p['speeds'][j]:.2f}")
                else:
                    parts.append(f"{NAMES[idx]}=X")
            info.append(f"#{p['id']}[h={int(p['h'])}]: " + " ".join(parts))
        out(f"[{frame_i}] kisiler={len(people)}")
        for line in info:
            out("   " + line)

        if len(people) >= 2:
            any_hit_this_frame = False
            for a, b in itertools.combinations(people, 2):
                ca, cb = a["center"], b["center"]
                d = float(np.linalg.norm(cb - ca))
                lim = config.FIGHT_NEAR * max(a["h"], b["h"])
                if d > lim:
                    if len(people) == 2:
                        n_mesafe += 1
                        out(f"   MESAFE_TOPLAM: d/h={d/max(a['h'], b['h']):.2f} "
                            f"esik={config.FIGHT_NEAR} (cok uzak duruyorsaniz normal)")
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
                        any_hit_this_frame = True
                        n_hit += 1
                        best_hit = max(best_hit, max(atk["speeds"]))
                        for hc in hit_cand:
                            out(f"   ATK#{atk['id']}->VIC#{vic['id']}: {hc}")
            if not any_hit_this_frame:
                n_hiz += 1
                mx = max((max(p["speeds"]) for p in people), default=0.0)
                out(f"   hic vurus sayilmadi (max_uzuv_hizi={mx:.2f}, esik={config.FIGHT_SPEED})")
        else:
            n_hiz += 1

    # tekrari azalt: en fazla 150 satir log tut, sonra ozet
    cap.release()
    out()
    out("=" * 60)
    out(f"OZET: toplam {frame_i} kare")
    out(f"  kisisiz kare: {n_kisi1} | 2+ kisi kare: {n_kisi2} | en fazla ayni karede: {n_en_fazla}")
    out(f"  MESAFE nedeniyle skip: {n_mesafe} kare")
    out(f"  vurus sayilmayan kare: {n_hiz}")
    out(f"  vurus SAYILAN kare: {n_hit} | en hizli uzuv: {best_hit:.2f}")
    out(f"  esikler: SPEED={config.FIGHT_SPEED} NEAR={config.FIGHT_NEAR} "
        f"MARGIN={config.FIGHT_MARGIN} POSE_CONF={config.POSE_CONF} KEY_CONF={key_conf}")
    out(f"  YORUM: {dict_interpret(n_kisi2, n_hit, n_mesafe, best_hit)}")

    dedup = []
    last = None
    for s in out_lines:
        if s == last:
            continue
        dedup.append(s)
        last = s
    log_path.write_text("\n".join(dedup), encoding="utf-8")
    print(f"Log: {log_path}")
    return dedup


def dict_interpret(n2, n_hit, n_mesafe, best):
    if n2 == 0:
        return "2 kisi HIC algilanmadi -> pose modeli goruyor mu / yeterli isik var mi?"
    if n_mesafe > 0 and n_mesafe >= n2 * 0.5:
        return "Kisiler cogunlukla 'sebebiyle skip' -> yaklasin veya FIGHT_NEAR artir"
    if best < config.FIGHT_SPEED and n_hit <= n2 * 0.02:
        return "Hiz hep esik alti -> yumruta daha sert/uzun kol hareketi ya da FIGHT_SPEED dusur"
    if n_hit == 0:
        return "Hiz yetiyor ama geometri/kutu reddediyor -> FIGHT_MARGIN artir"
    return "Kavga tetikleniyor; algilama calisiyor, gecikme/filtre ayari gerekebilir"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--seconds", type=int, default=10)
    args = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    video = TMP / "fight_rec.avi"
    log = TMP / "fight_diag.log"

    if video.exists():
        kullan = input(f"'{video.name}' mevcut. Yeniden kaydet? (y/n): ").strip().lower() != "n"
        if kullan:
            print("Yeniden kayit...")
            record(args.cam, args.seconds, video)
    else:
        print("Kayit yapiliyor...")
        record(args.cam, args.seconds, video)
    print("Analiz basliyor...")
    analyze(video, log, config.POSE_KEY_CONF)


if __name__ == "__main__":
    main()
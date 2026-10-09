import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAP_DIR = ROOT / "snapshots"
FRONTEND_DIR = ROOT / "frontend"

# --- Kameralar: 0 = webcam, "rtsp://..." = IP kamera, "demo.mp4" = video dosyası (döngüde oynar)
CAMERAS = {
    "cam1": {"name": "Giriş Kapısı", "source": 0},
    # "cam2": {"name": "Koridor A", "source": "rtsp://user:pass@192.168.1.20:554/stream"},
}

# --- Silah/bıçak modeli
_custom = ROOT / "models" / "weapon.pt"
MODEL_PATH = os.getenv("MODEL_PATH", str(_custom) if _custom.exists() else "yolov8n.pt")
CONF = float(os.getenv("CONF", 0.45))
IMGSZ = 640

CRITICAL_CLASSES = {"gun", "pistol", "handgun", "rifle", "firearm", "weapon", "shotgun", "knife_attack"}
WARNING_CLASSES = {"knife", "fight", "fighting", "violence", "scissors", "climb"}

# --- Eğitilmiş kavga/tırmanma modeli: models/fight.pt varsa otomatik kullanılır (sınıflar: climb, fight, normal)
_fm = ROOT / "models" / "fight.pt"
FIGHT_MODEL_PATH = os.getenv("FIGHT_MODEL_PATH", str(_fm) if _fm.exists() else "")
FIGHT_CONF = float(os.getenv("FIGHT_CONF", 0.5))

# --- Kavga / vurma tespiti (poz tahmini ile; ek model eğitmek gerekmez)
ENABLE_FIGHT = True
POSE_MODEL = "yolov8n-pose.pt"   # ilk çalıştırmada otomatik iner
FIGHT_IMGSZ = 480
FIGHT_SPEED = 4.5     # el/ayak hızı eşiği (vücut boyu / saniye). Düşür = daha hassas
FIGHT_NEAR = 1.0      # iki kişi arası mesafe sınırı (vücut boyu katı)
FIGHT_MIN_HITS = 2    # kavga için son WINDOW karede en az kaç tespit

# --- Yanlış alarm filtresi
WINDOW = 8
MIN_HITS = 3
COOLDOWN_SEC = 15

# --- Telegram (ortam değişkeniyle ver, koda yazma!)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "")  # örn. http://192.168.1.10:8000 (mesaja link eklenir)

# --- Alarm seviyesine göre aksiyon politikası
POLICY = {
    "warning":  {"telegram": True, "call_principal": False, "police": False, "escalate_after": 45},
    "critical": {"telegram": True, "call_principal": True,  "police": True,  "escalate_after": 0},
}
# warning: müdür paneline uyarı + Telegram. escalate_after sn içinde "Görüldü" denmezse otomatik KRİTİK'e yükselir.

POLICE_WEBHOOK_URL = os.getenv("POLICE_WEBHOOK_URL", "http://127.0.0.1:8000/mock/police")

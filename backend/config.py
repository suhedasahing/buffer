import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda *a, **k: False

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
SNAP_DIR = ROOT / "snapshots"
FRONTEND_DIR = ROOT / "frontend"
# 🥊 SADECE-KAVGA modu: diğer tüm tespitleri kapatmak için .env'ye ONLY_FIGHT=1 yaz.
# Varsayılan 0 = hepsi açık (silah/sigara/loitering/ROI/kavga).
ONLY_FIGHT = os.getenv("ONLY_FIGHT", "0") == "1"
# --- AKTİF ÖĞRENME (active learning): "yanlış alarm"/"doğrulandı" işaretlenen kareleri kaydet
DS_DIR = ROOT / "dataset"
ENABLE_ACTIVE_LEARNING = os.getenv("ACTIVE_LEARNING", "1") == "1"

# --- Kameralar: 0 = webcam, "rtsp://..." = IP kamera, "demo.mp4" = video dosyası (döngüde oynar)
CAMERAS = {
    "cam1": {"name": "Giriş Kapısı", "source": 0},
    # "cam2": {"name": "Koridor A", "source": "rtsp://user:pass@192.168.1.20:554/stream"},
}

# --- Silah/bıçak modeli
# v2 (tek 'weapon' sınıfı, 25k görselle eğitildi) varsa öncelikli; eski weapon.pt yedekte.
_custom = ROOT / "models" / "weapon_v2.pt"
if not _custom.exists():
    _custom = ROOT / "models" / "weapon.pt"
MODEL_PATH = os.getenv("MODEL_PATH", str(_custom) if _custom.exists() else "yolov8n.pt")
# Eşik dengesi: 0.45 uzaktaki silahı kaçırıyordu, 0.35 sahte alarm üretti; 0.40 ikisinin ortası.
# Asıl kazanç 960 imgsz + v2 model; eşiği zorlamak gerekmiyor.
CONF = float(os.getenv("CONF", 0.40))
# DEBUG: her çıkarımda ham tespitleri (düşük güvenliler dahil) konsola yaz -> DEBUG_DET=1
DEBUG_DET = os.getenv("DEBUG_DET", "0") == "1"
# FALSE POSITIVE FIX: NMS/IOU eşiği (aynı nesne için çoklu kutu üst üste binmesini bastırır).
IOU = float(os.getenv("IOU", 0.45))
# FALSE POSITIVE FIX: sınıf-bağımsız NMS ve maksimum kutu sayısı sınırı.
AGNOSTIC_NMS = os.getenv("AGNOSTIC_NMS", "1") == "1"
MAX_DET = int(os.getenv("MAX_DET", 20))
# UZAK/KÜÇÜK NESNE: çıkarım çözünürlüğü. Büyüdükçe uzaktaki silahı daha iyi görür ama YAVAŞLAR.
# Uzak silah tespiti 640'ta sık kaçtığı için 960'a çıkarıldı (CPU'da ~50ms, ~9-12FPS'lik pipeline yeterli).
# Sigara mesafeye (3-5m) yakın, donmayı önlemek için yeşil kendi imgsz'ini kullanır (GREEN_IMGSZ).
IMGSZ = int(os.getenv("IMGSZ", 960))
# ÇIKARIM TAVANI: model çıkarımı bu aralıktan sık çalışmaz -> CPU video akışına bırakılır.
INFER_INTERVAL = float(os.getenv("INFER_INTERVAL", 0.05))
# Kamera çözünürlüğü isteği (destekleyen webcam'lerde geniş açı). Zoom için CAP_PROP_ZOOM=0 denenir.
CAP_WIDTH = int(os.getenv("CAP_WIDTH", 1280))
CAP_HEIGHT = int(os.getenv("CAP_HEIGHT", 720))

# 🔴 Kırmızı alarm sınıfları (silah)
CRITICAL_CLASSES = {"gun", "pistol", "handgun", "rifle", "firearm", "weapon", "shotgun", "knife_attack"}
# 🟡 Sarı alarm sınıfları
WARNING_CLASSES = {"knife", "fight", "fighting", "violence", "scissors", "climb"}
# 🟢 Yeşil alarm sınıfları (sigara). weapon.pt'de sigara sınıfı YOK; bu yüzden ayrı model gerekir.
# "cigarettebutt": eğitilen models/cigarette.pt modelinin sınıf adı (izmarit tespiti).
GREEN_CLASSES = {"cigarette", "cigarettebutt", "cigarette_butt", "butt", "smoking", "smoke", "tobacco", "cig", "sigara"}
# Anahtar-kelime eşleşmesi: model sınıf adı ör. "Smoking - v5 Detectron2" olsa bile yakalar.
# NOT: models/cigarette.pt modelinde sınıflar bozuk etiketli; gerçek sigara/sigara-içme sınıfının
# adı "undefined" (ID 2). Test: sigara objesi 0.32, sigara içen kişi 0.63. Bu yüzden "undefined" eklendi.
GREEN_KEYWORDS = ("smok", "cig", "sigara", "tobacco", "tutun", "tütün", "butt", "vape", "undefined")

# --- 🟢 Sigara/yeşil alarm modeli: models/cigarette_v2.pt varsa otomatik yüklenir (opsiyonel)
_cig = ROOT / "models" / "cigarette_v2.pt"
if not _cig.exists():
    _cig = ROOT / "models" / "cigarette.pt"
CIGARETTE_MODEL_PATH = os.getenv("CIGARETTE_MODEL_PATH", str(_cig) if _cig.exists() else "")
if ONLY_FIGHT:
    CIGARETTE_MODEL_PATH = ""   # 🟢 sigara/kırmızı model kapalı: sadece kavga modu
# Sigara güven eşiği: 0.35 sabit sigarayı kaçırıyordu, 0.28 sahte alarm artırdı; 0.30 dengede.
GREEN_CONF = float(os.getenv("GREEN_CONF", 0.30))
# Yeşil kendi çözünürlüğü: silahtan ayrı tutulur, 640 yeterli (sigara yakın planda, 960'a gerek yok).
GREEN_IMGSZ = int(os.getenv("GREEN_IMGSZ", 640))

# --- Eğitilmiş kavga/tırmanma modeli (sınıflar: climb, fight, normal)
# UYARI: Mevcut models/fight.pt normal sahneyi "fight" sanıp çok yanlış pozitif (sarı alarm)
# ürettiği için VARSAYILAN OLARAK KAPATILDI. Kavga tespiti aşağıdaki poz (pose) dedektörüyle yapılır.
# fight.pt'nizi düzeltip tekrar açmak isterseniz: ortam değişkeni verin -> FIGHT_MODEL_PATH=models/fight.pt
FIGHT_MODEL_PATH = os.getenv("FIGHT_MODEL_PATH", "")
_fm = ROOT / "models" / "fight.pt"
FIGHT_CONF = float(os.getenv("FIGHT_CONF", 0.70))

# --- Kavga / vurma tespiti (poz tahmini ile; ek model eğitmek gerekmez)
ENABLE_FIGHT = True
POSE_MODEL = "yolov8n-pose.pt"   # ilk çalıştırmada otomatik iner
# Kişi tespit eşiği: 0.35 iken ikinci kişi (uzak/kısmen örtülü) takip kimliği alamıyor,
# kavga mantığı hiç devreye girmiyordu (log: kisiler=1). 0.25 ile iki kişi de yakalanır.
POSE_CONF = float(os.getenv("POSE_CONF", 0.25))
# UZAK/KÜÇÜK KİŞİ: çıkarım çözünürlüğü. 640->960 küçük (uzaktaki) kişilerin el/ayak
# anahtar noktalarını belirgin biçimde iyileştirir (CPU'da poz 640~31ms, 960~62ms).
# CPU'su dar olan sistemde FIGHT_IMGSZ=640 ile geri alın; hafif gecikme pahasına 960 önerilir.
FIGHT_IMGSZ = int(os.getenv("FIGHT_IMGSZ", 960))
# El hizi esigi (vucut boyu / saniye). Canli oleumlerde bu mesafedeki (3-5m) gercek yumruk
# hizlari 0.5-1.45 arasi cikiyordu; 2.2/1.6 hep yuksek kaliyordu. 1.1'e duserken skorlama KOL
# sinirlidir (bilek+dirsek); ayaklar yuruyusle 1.6+ olup sahte alarm urettigi icin sayilmaz.
# Yuruyuste kol sallanisi ~0.3-0.6, yumruk ~1.0-1.5 -> ayisim net.
# 0.9: canli demoda yumruklar 1.1-2.5 arasindaydi ve bazi denemeler eşiğin altında kayboluyordu.
FIGHT_SPEED = float(os.getenv("FIGHT_SPEED", 0.9))
# İki kişi arası mesafe sınırı (vücut boyu katı). Büyüt = biraz uzakta olanları da değerlendirir.
FIGHT_NEAR = float(os.getenv("FIGHT_NEAR", 1.6))
# Anahtar noktası güven eşiği: hızlı yumrukta bilek noktası motion blur yüzünden çoğu karede
# 0.2'nin altına düşüp hız 0 sayılıyordu (log: maks_uzuv_hizi~0.94). 0.1 ile gerçek hareket
# yakalanır; yine de FIGHT_SPEED + geometri (karşıya doğru) yanlış alarmı bastırır.
POSE_KEY_CONF = float(os.getenv("POSE_KEY_CONF", 0.1))
# Vuruş geometrisi: karşı kişinin kutusu iki merkez arası mesafenin bu katı kadar genişletilir.
# 0.5 => kol, kişiler arasındaki boşluğun ORTASINI (midline) geçtiyse vuruş sayılır. Böylece
# kutuya hiç girmeyen ama karşıya doğru savrulan yumruklar da yakalanır.
# 1.0 => kutu belirgin biçimde geniş: canlı demoda (yumuşak/esame yumruklar, hız 1.5-2.5)
# usturuk karşı kutuya nadiren derinlemesine giriyor ve skor hep 0.00 kalıyordu; genişletince
# kompakt yumruklar da sayılır (sahte-alarm riski biraz artar, seviye yalnızca sarı/uyarıdır).
FIGHT_MARGIN = float(os.getenv("FIGHT_MARGIN", 1.0))
FIGHT_MIN_HITS = 1    # kavga: bu pencerede kaç kare "vuruş" sayılmalı (canlıda tek vuruş da alarm; cooldown zaten 15sn)

# --- 🟡 ŞÜPHELİ BEKLEŞ (loitering): tek bir kişi uzun süre sahnede kalıp HAREKETSİZ durursa
ENABLE_LOITERING = os.getenv("ENABLE_LOITERING", "1") == "1"
LOITERING_TIME = float(os.getenv("LOITERING_TIME", 10.0))      # sahnede kaç saniye kalınca aday sayılır
LOITERING_WINDOW = float(os.getenv("LOITERING_WINDOW", 5.0))   # bu süre boyunca uyarı üretir (sonra durur)
LOITERING_STILL_SPEED = float(os.getenv("LOITERING_STILL_SPEED", 0.35))  # uzuv hızı bu altı = "hareketsiz" (vücut boyu/sn)
LOITERING_STILL_MOVE = float(os.getenv("LOITERING_STILL_MOVE", 0.06))    # merkez kayması bu altı = "yerinde" (vücut boyu oranı)
# DEBUG: kavga/poz çıkarımında kişi sayısı + en yüksek uzuv hızını konsola yaz -> DEBUG_FIGHT=1
DEBUG_FIGHT = os.getenv("DEBUG_FIGHT", "0") == "1"

# --- 🔴 İntihar / camdan atlama: SANAL "Pencere/Tehlikeli Bölge" (ROI) ayarları
# ROI_ZONE, kare boyutuna göre NORMALİZE edilmiş (x1, y1, x2, y2) kutusudur (0.0 - 1.0).
# Varsayılan: karenin sağ-üst köşesindeki pencere/korkuluk alanı. KENDİ KAMERANIZA GÖRE KALİBRE EDİN.
# Latch (kilit) sayesinde kişi bölgede sürekli durursa alarm BİR kez çalar; bölgeden çıkınca tekrar kurulur.
ROI_ENABLED = os.getenv("ROI_ENABLED", "1") == "1"
# Varsayılan: karenin sağ-üst KÖŞESİNDEKİ dar pencere bandı. Kişinin merkezi bu banda girmeli.
ROI_ZONE = tuple(float(v) for v in os.getenv("ROI_ZONE", "0.72,0.0,1.0,0.38").split(","))
ROI_MIN_HITS = int(os.getenv("ROI_MIN_HITS", 4))   # kişi merkezi bölgede kaç ardışık kare kalırsa alarm
DROP_RATIO = float(os.getenv("DROP_RATIO", 0.60))  # ani y ekseni düşüşü eşiği (vücut boyu oranı)
DROP_MIN_HITS = int(os.getenv("DROP_MIN_HITS", 3)) # ani düşüş kaç ardışık karede doğrulanırsa alarm
# --- 🔴 Tırmanma (intihar/kaçış): kişi karenin üst kısmındayken YUKARI hareket ediyorsa
RISE_RATIO = float(os.getenv("RISE_RATIO", 0.06))    # kare başına yukarı hareket eşiği (vücut boyu oranı)
RISE_MIN_HITS = int(os.getenv("RISE_MIN_HITS", 4))   # kaç ardışık kare yukarı hareket doğrulanırsa alarm
CLIMB_TOP = float(os.getenv("CLIMB_TOP", 0.60))      # kişi merkezi karenin üst %X'inde olmalı (0.60 = üst %60)

# 🥊 SADECE-KAVGA modu uygulamaları: loitering ve ROI/intihar/atlama da kapatılır.
if ONLY_FIGHT:
    ENABLE_LOITERING = False
    ROI_ENABLED = False

# --- Performans: FRAME SKIPPING (aynı döngüde tüm modelleri her karede çalıştırma)
PROCESS_EVERY = int(os.getenv("PROCESS_EVERY", 2))  # ana silah modeli her N karede bir çalışır
GREEN_EVERY = int(os.getenv("GREEN_EVERY", 1))       # 🟢 sigara modeli her N turda bir çalışır (1 = her tur, hızlı)
FIGHT_EVERY = int(os.getenv("FIGHT_EVERY", 1))       # kavga/poz modeli her N karede bir çalışır (1 = her tur, hızlı)

# --- Yanlış alarm filtresi
WINDOW = 8
MIN_HITS = 3
CRIT_MIN_HITS = int(os.getenv("CRIT_MIN_HITS", 2))  # kırmızı için ardışık çıkarım sayısı (2 = hızlı)
GREEN_MIN_HITS = int(os.getenv("GREEN_MIN_HITS", 2))  # sigara: bu pencerede en az kaç vuruş
# Sigara ONAY PENCERESİ: hareketli/kısa görünümlerde tespit kesikli gelir. Kısa pencere
# (WINDOW=8) bunları kaçırır; bu yüzden sigaraya özel geniş pencere kullanılır.
GREEN_WINDOW = int(os.getenv("GREEN_WINDOW", 20))
# ⚡ ANINDA TETİKLEME: bu güvenin ÜZERİNDEKİ tespit, vuruş birikmesini BEKLEMEDEN alarm üretir (hız).
IMMEDIATE_CONF = float(os.getenv("IMMEDIATE_CONF", 0.65))
# 🧷 STABİLİTE: tespit kısa süre kesilse bile durumu bu kadar sn koru (ekranda/alarmda titreme olmasın).
STATE_HOLD_SEC = float(os.getenv("STATE_HOLD_SEC", 1.5))
COOLDOWN_SEC = 15

# --- Telegram (repo kökündeki .env dosyasından okunur; koda YAZILMA!)
# Şablon için .env.example'a bak. Token: @BotFather ile bot oluştur; Chat ID: get_chat_id.py ile al.
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "")  # örn. http://192.168.1.10:8000 (mesaja link eklenir)

# --- Alarm seviyesine göre aksiyon politikası (3 kademeli)
# sound: "none"=ses yok, "warning"=kısa çift bip, "siren"=sürekli siren
POLICY = {
    # 🟢 YEŞİL (Sigara): sesli uyarı YOK. Sadece müdür paneline sessiz bildirim/log.
    "green":    {"telegram": False, "call_principal": False, "police": False,
                 "police_ready": False, "escalate_after": 0,  "sound": "none"},
    # 🟡 SARI (Kavga): bilgisayarda uyarı sesi + müdür paneline bildirim (+Telegram).
    "warning":  {"telegram": True,  "call_principal": False, "police": False,
                 "police_ready": False, "escalate_after": 45, "sound": "warning"},
    # 🔴 KIRMIZI (Silah / İntihar·camdan atlama): siren + müdür + polise bildirim
    #    + arayüzdeki manuel "Polise Bildir" butonunu aktifleştiren police_ready flag'i.
    "critical": {"telegram": True,  "call_principal": True,  "police": True,
                 "police_ready": True,  "escalate_after": 0,  "sound": "siren"},
}
# warning: müdür paneline uyarı + Telegram. escalate_after sn içinde "Görüldü" denmezse otomatik KRİTİK'e yükselir.

POLICE_WEBHOOK_URL = os.getenv("POLICE_WEBHOOK_URL", "http://127.0.0.1:8000/mock/police")

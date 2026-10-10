"""Telegram bildirimi. TELEGRAM_TOKEN ve TELEGRAM_CHAT_ID ortam değişkeni olarak verilir."""
import time
from pathlib import Path
import httpx
from . import config

TITLES = {"green": "🟢 GÖZLEM", "warning": "🟡 POTANSİYEL TEHDİT", "critical": "🔴 KRİTİK TEHDİT"}
NAMES = {"gun": "Silah", "handgun": "Tabanca", "rifle": "Tüfek", "knife": "Bıçak",
         "fight": "Kavga / vurma", "climb": "Tırmanma / izinsiz giriş", "scissors": "Makas",
         "suicide": "İntihar / camdan atlama", "cigarette": "Sigara", "loitering": "Şüpheli bekleyiş",
         "smoking": "Sigara içme", "smoke": "Sigara/duman"}


def configured() -> bool:
    return bool(config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID)


async def send_telegram(ev: dict, extra: str = "") -> dict:
    if not configured():
        print("[telegram] ATLANDI: TELEGRAM_TOKEN / TELEGRAM_CHAT_ID tanımlı değil")
        return {"status": "skipped", "reason": "TELEGRAM_TOKEN / TELEGRAM_CHAT_ID tanımlı değil"}
    text = (f"{TITLES.get(ev['level'], 'UYARI')}\n"
            f"Kamera: {ev['camera_name']}\n"
            f"Tespit: {NAMES.get(ev['label'], ev['label'])} (%{round(ev['conf'] * 100)})\n"
            f"Saat: {time.strftime('%H:%M:%S', time.localtime(ev['ts'] / 1000))}")
    if extra:
        text += f"\n{extra}"
    if config.DASHBOARD_URL:
        text += f"\nPanel: {config.DASHBOARD_URL}"
    base = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}"
    snap = config.SNAP_DIR / Path(ev.get("snapshot", "")).name
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            if snap.is_file():
                r = await c.post(f"{base}/sendPhoto", data={"chat_id": config.TELEGRAM_CHAT_ID, "caption": text},
                                 files={"photo": (snap.name, snap.read_bytes(), "image/jpeg")})
            else:
                r = await c.post(f"{base}/sendMessage", json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text})
            j = r.json()
            if j.get("ok"):
                print("[telegram] gönderildi")
                return {"status": "sent"}
            print("[telegram] HATA:", j.get("description"))
            return {"status": "failed", "error": j.get("description", "bilinmiyor")}
    except Exception as e:
        print("[telegram] HATA:", type(e).__name__)
        return {"status": "failed", "error": type(e).__name__}

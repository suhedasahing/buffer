"""Tamamen SİMÜLASYON: gerçek arama/polis entegrasyonu yok."""
import asyncio, uuid
import httpx
from . import config


async def call_principal(ev: dict) -> dict:
    await asyncio.sleep(1.0)  # arama çalıyor simülasyonu
    print(f"[SIM-CALL] Müdür aranıyor -> {ev['camera_name']} / {ev['label']}")
    return {"status": "calling", "to": "Okul Müdürü (+90 5XX XXX XX XX)"}


async def notify_police(ev: dict) -> dict:
    payload = {
        "type": "SCHOOL_THREAT", "level": ev["level"], "label": ev["label"],
        "camera": ev["camera_name"], "ts": ev["ts"], "snapshot": ev["snapshot"],
        "school": "Demo Okulu", "address": "Örnek Mah. Okul Sk. No:1",
    }
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.post(config.POLICE_WEBHOOK_URL, json=payload)
            return {"status": "sent", "response": r.json()}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def mock_police_receive(payload: dict) -> dict:
    ref = uuid.uuid4().hex[:8].upper()
    print(f"[MOCK-POLICE] Bildirim alındı ref={ref}: {payload}")
    return {"received": True, "ref": ref, "eta_min": 6}

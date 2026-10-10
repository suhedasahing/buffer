import asyncio, time, uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import config, alerts, notifier, sound, active_learning
from .detector import ThreatDetector
from .camera import CameraWorker

config.SNAP_DIR.mkdir(exist_ok=True)
WORKERS: dict[str, CameraWorker] = {}
LOGS: list[dict] = []
CLIENTS: set[WebSocket] = set()
LOOP: asyncio.AbstractEventLoop | None = None
HANDLED = {"ack", "principal_called_manual", "police_notified_manual"}  # bunlar yükseltmeyi durdurur


async def broadcast(msg: dict):
    dead = []
    for ws in list(CLIENTS):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        CLIENTS.discard(ws)


def find(ev_id):
    return next((e for e in LOGS if e["id"] == ev_id), None)


async def log_action(ev_id: str, action: str, detail):
    ev = find(ev_id)
    status = None
    if ev:
        ev["actions"].append({"action": action, "detail": detail, "ts": int(time.time() * 1000)})
        if action == "false_alarm":
            ev["status"] = "false_alarm"
            active_learning.save(ev, "negative")   # aktif öğrenme: negatif örnek
        elif action in HANDLED and ev["status"] == "open":
            ev["status"] = "ack"
            active_learning.save(ev, "positive")   # aktif öğrenme: doğrulanmış pozitif örnek
        status = ev["status"]
    await broadcast({"type": "action", "event_id": ev_id, "action": action, "detail": detail, "status": status})
    # 🔴 Siren, alarm "görüldü" veya "yanlış alarm" işaretlenince susturulur.
    if action in ("ack", "false_alarm"):
        sound.stop_siren()


async def step(ev_id: str, action: str, coro):
    await log_action(ev_id, action, await coro)


async def run_policy(ev: dict):
    """Seviyeye göre otomatik aksiyonlar (config.POLICY) + bilgisayar sesi."""
    p = config.POLICY[ev["level"]]
    # 🔊 3 kademeli ses: yeşil=sessiz, sarı=kısa uyarı sesi, kırmızı=siren
    mode = p.get("sound", "none")
    if mode == "warning":
        sound.warning()
    elif mode == "siren":
        sound.start_siren()
    if not p["call_principal"]:
        await log_action(ev["id"], "principal_notified", "Müdür paneline uyarı düştü")
    if p["telegram"]:
        asyncio.create_task(step(ev["id"], "telegram_sent", notifier.send_telegram(ev)))
    if p["call_principal"]:
        asyncio.create_task(step(ev["id"], "principal_called", alerts.call_principal(ev)))
    if p["police"]:
        asyncio.create_task(step(ev["id"], "police_notified", alerts.notify_police(ev)))


async def escalation_timer(ev: dict, delay: int):
    await asyncio.sleep(delay)
    if ev["status"] == "open" and ev["level"] == "warning":
        ev["level"], ev["escalated"] = "critical", True
        ev["police_ready"] = config.POLICY["critical"].get("police_ready", True)
        await broadcast({"type": "escalated", "event": ev})
        await log_action(ev["id"], "escalated", f"{delay} sn içinde onaylanmadı, KRİTİK'e yükseltildi")
        await run_policy(ev)


async def handle_event(ev: dict):
    ev["id"], ev["status"], ev["actions"], ev["escalated"] = uuid.uuid4().hex[:8], "open", [], False
    # 🔴 "Polise Bildir" butonunu aktifleştiren veri flag'i
    ev["police_ready"] = config.POLICY[ev["level"]].get("police_ready", False)
    LOGS.insert(0, ev)
    del LOGS[200:]
    await broadcast({"type": "alert", "event": ev})
    await run_policy(ev)
    delay = config.POLICY[ev["level"]].get("escalate_after", 0)
    if delay:
        asyncio.create_task(escalation_timer(ev, delay))


def on_event_threadsafe(ev: dict):
    asyncio.run_coroutine_threadsafe(handle_event(ev), LOOP)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global LOOP
    LOOP = asyncio.get_running_loop()
    detector = ThreatDetector()
    active_learning.register("weapon", detector.model_names)
    # 🟡 Sarı alarm: eğitilmiş kavga/tırmanma modeli (fight.pt) varsa yükle.
    fight_model = None
    if config.FIGHT_MODEL_PATH:
        fight_model = ThreatDetector(config.FIGHT_MODEL_PATH, config.FIGHT_CONF, source="fight_model")
        active_learning.register("fight_model", fight_model.model_names)
    for cid, cfg in config.CAMERAS.items():
        w = CameraWorker(cid, cfg, detector, on_event_threadsafe, fight_model)
        WORKERS[cid] = w
        w.start()
    yield
    for w in WORKERS.values():
        w.running = False


app = FastAPI(title="XNOR Guard", lifespan=lifespan)
app.mount("/snapshots", StaticFiles(directory=config.SNAP_DIR), name="snapshots")


@app.get("/")
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")


@app.get("/api/cameras")
def cameras():
    return [{"id": k, "name": v.name_, "state": v.state, "fps": round(v.fps, 1)} for k, v in WORKERS.items()]


@app.get("/api/logs")
def logs():
    return LOGS


@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: str):
    w = WORKERS.get(cam_id)
    if not w:
        raise HTTPException(404, "kamera yok")

    async def gen():
        while True:
            jpg = w.get_jpeg()
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(0.04)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


class ActionReq(BaseModel):
    event_id: str
    action: str  # ack | call_principal | call_police | telegram | false_alarm


@app.post("/api/action")
async def manual_action(req: ActionReq):
    ev = find(req.event_id)
    if not ev:
        raise HTTPException(404, "olay yok")
    if req.action == "call_principal":
        res, name = await alerts.call_principal(ev), "principal_called_manual"
    elif req.action == "call_police":
        res, name = await alerts.notify_police(ev), "police_notified_manual"
    elif req.action == "telegram":
        res, name = await notifier.send_telegram(ev, "(manuel gönderim)"), "telegram_manual"
    elif req.action in ("false_alarm", "ack"):
        res, name = "Güvenlik görevlisi işaretledi", req.action
    else:
        raise HTTPException(400, "bilinmeyen aksiyon")
    await log_action(ev["id"], name, res)
    return {"ok": True}


class SettingsReq(BaseModel):
    conf: float | None = None
    fight_speed: float | None = None


@app.get("/api/settings")
def get_settings():
    return {"conf": config.CONF, "fight_speed": config.FIGHT_SPEED, "telegram": notifier.configured(),
            "fight_enabled": config.ENABLE_FIGHT}


@app.post("/api/settings")
def set_settings(s: SettingsReq):
    if s.conf is not None:
        config.CONF = max(0.25, min(0.95, s.conf))  # gerçek silahı da yakalayacak aralık
    if s.fight_speed is not None:
        config.FIGHT_SPEED = max(1.0, min(15.0, s.fight_speed))
    return get_settings()


@app.post("/api/simulate")
async def simulate(level: str = "critical", camera: str = "cam1", label: str = ""):
    w = WORKERS.get(camera)
    if not w or level not in ("green", "warning", "critical"):
        raise HTTPException(400, "camera/level hatalı")
    default = {"critical": "gun", "warning": "fight", "green": "cigarette"}[level]
    w.emit(level, label or default, 0.91)
    return {"ok": True}


@app.post("/mock/police")
async def mock_police(payload: dict):
    return alerts.mock_police_receive(payload)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        CLIENTS.discard(ws)

"""Live pipeline testi: fight_rec.avi'yi GERCEL canli uygulamanin CameraWorker'i
ile besler. Uygulama kavga arastirilirkEN neyin alarm urettigini/produrmadigini gosterir.

Kullanım: .venv\\Scripts\\python.exe training\\pipeline_test.py
"""
import sys
import time
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.camera import CameraWorker  # noqa: E402
from backend.detector import ThreatDetector  # noqa: E402
from backend import active_learning  # noqa: E402

VIDEO = Path("C:/Users/suhed/AppData/Local/Temp/opencode/fight_rec.avi")

events = []
lock = threading.Lock()


def on_event(ev):
    with lock:
        events.append(ev)
        print(f"[EVENT] {ev['level']:8s} {ev['label']:12s} conf={ev['conf']} source={ev.get('source')}")


def main():
    cfg = {"name": "TEST", "source": str(VIDEO)}
    det = ThreatDetector()
    worker = CameraWorker(0, cfg, det, on_event, fight_model=None)
    worker.name_ = "TEST"
    worker.start()
    t0 = time.time()
    while time.time() - t0 < 20:
        time.sleep(0.5)
        with lock:
            active_ev = [e for e in events
                         if e["level"] != "green" or time.time() - e["ts"] / 1000 < 30]
        if any(e["label"] == "fight" for e in active_ev):
            break
    worker.running = False
    time.sleep(0.5)
    with lock:
        stat = sorted(f"{e['level']}:{e['label']}" for e in events)
    print("=" * 60)
    print(f"Kosu suresi: {time.time()-t0:.1f}s, toplam olay: {len(events)}")
    for kind in sorted(set(stat)):
        print(f"  {kind}: {stat.count(kind)}")
    f_sim = [e for e in events if e["label"] == "fight"]
    print("KAVGA ALARMI:", "VAR ✓" if f_sim else "YOK ✗")
    if not f_sim:
        print("-> canli hattaki sorun tespit edildi, camera.py/_decide incelenmeli")
    elif len(events) <= 2:
        print("-> tititi ama filtre/cooldown sebebiyle cok az: FIGHT_MIN_HITS/COOLDOWN mi?")


if __name__ == "__main__":
    main()
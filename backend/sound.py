"""Sunucu (bilgisayar) tarafı sesli uyarı modülü.

Windows'ta stdlib `winsound` kullanılır; `winsound` yoksa (Linux/Mac) konsol zili
(`os.write(1, b"\\a")`) yedek olarak çalışır. Bu modül hem 🟡 alarm için kısa bir
çift "bip", hem de 🔴 alarm için kesintisiz bir "siren" sağlar.

Tüm sesler ayrı thread'lerde çalışır; böylece kamera döngüsü (inference) bloke olmaz.
"""
import os
import sys
import time
import threading

try:
    import winsound  # Windows stdlib
except Exception:  # diğer platformlar
    winsound = None

_lock = threading.Lock()
_siren_stop = threading.Event()
_siren_thread = None


def _beep(freq: int, dur_ms: int):
    """Tek bir ton çalar. winsound varsa donanım bip'i, yoksa konsol zili."""
    if winsound:
        try:
            winsound.Beep(int(freq), int(dur_ms))
            return
        except Exception:
            pass
    try:
        os.write(1, b"\a")
        sys.stdout.flush()
    except Exception:
        pass


def warning():
    """🟡 Sarı alarm: kısa çift bip (sesli uyarı)."""
    def _run():
        _beep(660, 180)
        time.sleep(0.12)
        _beep(660, 180)
    threading.Thread(target=_run, daemon=True).start()


def _siren_loop():
    """🔴 Siren: durdurulana kadar iki ton arasında gidip gelen kesintisiz ses."""
    while not _siren_stop.is_set():
        _beep(880, 280)
        if _siren_stop.is_set():
            break
        _beep(560, 280)


def start_siren(duration: int = 30):
    """🔴 Kırmızı alarm sirenini başlatır (varsayılan 30 sn sonra otomatik susar)."""
    global _siren_thread
    with _lock:
        if _siren_thread and _siren_thread.is_alive():
            return
        _siren_stop.clear()
        _siren_thread = threading.Thread(target=_siren_loop, daemon=True)
        _siren_thread.start()
    # Güvenlik ağı: süre dolunca otomatik durdur (görüldü/yanlış alarm ile de erken durur).
    threading.Timer(duration, stop_siren).start()


def stop_siren():
    """Sireni durdurur (alarm 'görüldü' / 'yanlış alarm' işaretlenince çağrılır)."""
    _siren_stop.set()

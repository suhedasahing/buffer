"""Aktif öğrenme: panelde "yanlış alarm" / "doğrulandı" denilen kareleri YOLO veri setine kaydeder.

Kaydedilen yapı (model kaynağına göre ayrılır; ör. source=weapon / cigarette / fight_model):

    dataset/<source>/
        images/  *.jpg
        labels/  *.txt        (YOLO formatı: "cls cx cy w h"; negatiflerde boş dosya)
        classes.json          (sınıf adları, ilk kayıtta yazılır)
        data.yaml             (yeniden eğitim için hazır)

Negatifler (yanlış alarm) boş etiketle yazılır -> model "burada nesne yok" öğrenir ve
yanlış pozitifler zamanla azalır. Yeniden eğitim:  yolo train data=dataset/<source>/data.yaml ...
"""
import json
import shutil
import time
from pathlib import Path

from . import config

# source -> sınıf adları listesi (model yüklenirken register edilir)
_META: dict[str, list] = {}


def register(source: str, names):
    if source and names:
        _META[source] = list(names)


def _snapshot(ev: dict) -> Path:
    return config.SNAP_DIR / Path(ev.get("snapshot", "")).name


def _write_meta(base: Path, source: str):
    names = _META.get(source)
    if not names:
        return
    (base / "classes.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"path: {base.as_posix()}", "train: images", "val: images", "names:"] + \
            [f"  {i}: {n}" for i, n in enumerate(names)]
    (base / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save(ev: dict, kind: str) -> str | None:
    """kind: 'negative' (yanlış alarm) | 'positive' (doğrulandı). Kaydedilen dosya adını döner."""
    if not config.ENABLE_ACTIVE_LEARNING:
        return None
    src = _snapshot(ev)
    if not src.is_file():
        return None

    source = ev.get("source") or "weapon"
    base = config.DS_DIR / source
    img_dir, lbl_dir = base / "images", base / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    name = f"{kind}_{ev.get('level', 'na')}_{ev.get('label', 'na')}_{ev.get('id', 'x')}_{int(time.time() * 1000)}"
    shutil.copy2(src, img_dir / f"{name}.jpg")

    box, cls = ev.get("box"), ev.get("cls")
    if kind == "positive" and box and cls is not None:
        cx, cy, w, h = box  # normalize edilmiş (0-1)
        (lbl_dir / f"{name}.txt").write_text(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    else:
        (lbl_dir / f"{name}.txt").write_text("")  # negatif: nesne yok

    _write_meta(base, source)
    print(f"[active] {kind} kaydedildi -> {base}\\images\\{name}.jpg")
    return str(img_dir / f"{name}.jpg")

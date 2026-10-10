# xnor-guard Egitim Runbook (Colab T4)

Tum veri Kaggle/Roboflow'dan dogrudan Colab icinde cekildigi icin PC'ye GB yuklenmez.
Tek istisna: `training/` paketi (script + config'ler) Yuklenir.

## 0. Gerekli anahtarlar (repo'ya YAZILMA, Colab secrets / env olarak girilir)

- Kaggle **username**: `suwasted` (profil URL slug'i, dogrulandi)
- Kaggle key: `KGAT_...`
- Roboflow key: `76d8bnuMOrFGslzhsihk`

Colab'de: sol raf > Key (lock) > "+ New secret" ile `KAGGLE_USERNAME`, `KAGGLE_KEY`, `RF_KEY` ekle.

## 1. Repo paketini yukle + bagimliliklar

```python
# training klasorunu (script+config, data HARIC) /content'e zip'leyip yukle, sonra:
!pip install -q ultralytics roboflow kaggle
import os
from google.colab import userdata
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
os.environ["RF_KEY"] = userdata.get("RF_KEY")
```

## 2. Veri (Kaggle + Roboflow) -> /content/datasets

```bash
!mkdir -p /content/datasets
!kaggle datasets download -d apex25/weapons-detection-raw -p /content/datasets
!unzip -q /content/datasets/weapons-detection-raw.zip -d /content/datasets/weapons-detection-raw
```

Roboflow setleri (SDK ile, `overwrite=True` sart — yoksa klasor varsa atlar):
```python
import os
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["RF_KEY"])
for ws, proj, ver, out in [
    ("new-workspace-pnwnb", "gun-detect-y1ogv", 1, "/content/Gun-Detect-1"),
    ("joao-assalim-xmovq", "weapon-2", 2, "/content/datasets/weapon-2"),
]:
    rf.workspace(ws).project(proj).version(ver).download("yolov8", location=out, overwrite=True)
```

Not: `weapons-detection-raw` zip icinde klasor yapisini dogrula; config path'i buna gore
`training/weapon_sources.json` icinde ayarla (istege bagli).

## 3. Sigara verisi (Roboflow, Colab icinden)

```python
for ws, proj, ver, out in [
    ("smoking-pwchg", "cigarette-smokers", 1, "/content/datasets/cigarette-smokers"),
    ("objectdetection-rrcdv", "smoking-detection-dataset", 4, "/content/datasets/smoking-detection-dataset"),
]:
    rf.workspace(ws).project(proj).version(ver).download("yolov8", location=out, overwrite=True)
```

## 4. Harman (mevcut script, degismeden calisir)

```bash
# calisma dizini: /content (training/ klasoru ve datasets/ burada olmali)
!python training/prepare_dataset.py --config training/weapon_sources.json \\
    --out training/data/weapon_v2 --class-name weapon
!python training/prepare_dataset.py --config training/smoking_sources.json \\
    --out training/data/cigarette_v2 --class-name cigarette
```

report.json'da img/kutu sayilarini dogrula (ufak sapma normal: min-box ve sinif filtreleri).

## 5. Egitim (T4)

```bash
!python training/train_yolo.py \\
    --data training/data/weapon_v2/data.yaml \\
    --model yolov8s --epochs 80 --imgsz 640 \\   # uzak hedef onceligi: --imgsz 800 dene (fps duser)
    --batch -1 --device 0 --name weapon_v2
!python training/train_yolo.py \\
    --data training/data/cigarette_v2/data.yaml \\
    --model yolov8s --epochs 60 --imgsz 640 \\
    --batch -1 --device 0 --name cigarette_v2
```

- Cikti: `runs/weapon_v2/weights/best.pt`, `runs/cigarette_v2/weights/best.pt`
- `runs/<name>/training_summary.json`: f1/mAP ozet.
- Epoch'lar dengesiz ise patience erken durdurur (auto).

## 6. indirme

```python
from google.colab import files
files.download("/content/xnor-guard/runs/weapon_v2/weights/best.pt")
files.download("/content/xnor-guard/runs/cigarette_v2/weights/best.pt")
```

PC'de `models/weapon_v2.pt` ve `models/cigarette_v2.pt` olarak adlandir (eski .pt'ler korunur).
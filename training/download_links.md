# Veri seti kaynaklari ve indirme komutlari

Tumu CC BY 4.0 / Kaggle lisansli. Indirdikten sonra config JSON'lari guncelle.

## SILAH (tek sinif "weapon")

### 1. Kaggle - weapons-detection-raw (ONCELIKLI, en temiz)
- 20,010 gorsel, 33,224 kutu, 6 sinif (pistol, rifle, knife, grenade, heavy_gun, person). ~2.1 GB.
- `kaggle datasets download -d apex25/weapons-detection-raw`
- Ac: `unzip weapons-detection-raw.zip -d ../datasets/weapons-detection-raw`
- Sinif id sira: 0=pistol 1=rifle 2=knife 3=grenade 4=heavy_gun 5=person (person AYRILIR). `weapon_sources.json`'da bu map hazir.
- Dizin duzeni: `<root>/train/{images,labels}`, `<root>/val/...`, `<root>/test/...` (eger `images/train` duzenindeyse script icin duzenle).

### 2. Roboflow - weapon-2 (knife + guns; pistol/rifle ayrimi yok)
- https://universe.roboflow.com/joao-assalim-xmovq/weapon-2/dataset/2 (4,098 gorsel)
- Indirme: Roboflow sayfasinda "Download Dataset" -> YOLOv8 -> zip `weapon-2.v1i.yolov8.zip` indir.
- `unzip weapon-2*.zip -d ../datasets/weapon-2`
- Sinif sira: 0=knife 1=guns -> her ikisi -> weapon.

### 3. Lokal - Gun-Detect-1 (dokunulmaz)
- `Gun-Detect-1/` zaten repoda (1863 gorsel). Bagimsiz dis test olarak tutmak istersen
  `prepare_dataset.py` calistirirken config'de bu kaynagi ayri tut; eger test seti olarak
  kullanacaksan `--config` yerine disarida ayri bir config (yalniz gun_detect_1) uret.

## KAVGA (video klipleri)

### 1. airtlab (ONCELIKLI - hem violent hem "sarilma/alkis" hard-negatif, 1080p30)
- https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos
- `git clone https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos repo`
- 350 klip: `violent/{cam1,cam2}`, `non-violent/{cam1,cam2}` (avg 5.6 sn).
- `python training/prepare_fight_videos.py --root repo/violence-detection-dataset --out training/data/fight_v1`

### 2. CCTV-Fights (Kaggle, gercek CCTV kavgasi)
- https://www.kaggle.com/datasets/shreyj1729/cctv-fights-dataset
- `kaggle datasets download -d shreyj1729/cctv-fights-dataset`
- Hem CCTV hem NON-CCTV + train/test/val klasorleri var. Oneri: CCTV + NON-CCTV kavga kliplerini tek klasore kopyala,
  `--csv` ile label (1) ver.

### 3. RWF-2000 (Kaggle uzerinden 2000 klip)
- Kaggle'da `yash07yadav/project-data` (Violence Detection - Combined) Fight/NonFight klipleri icerir.
- `kaggle datasets download -d yash07yadav/project-data`
- Fight/NonFight klasorlerine gore `--root` kullan.

### 4. Hockey Fight (1000 klip: 500 kavga / 500 degil) - istege bagli
- Kaggle aynasi: `kaggle datasets download -d yassershrief/hockey-fight-vidoes`
- `--root` altinda fi_fight / fi_nonfight.

## SIGARA ("ağızda sigara" hedefi)

### 1. Roboflow - cigarette_detection (873 gorsel, "cigarette")
- https://universe.roboflow.com/cigarettedetection/cigarette_detection -> Download -> YOLOv8 -> `..\datasets\cigarette_detection`

### 2. Roboflow - smoking detection dataset (2756 gorsel, cigarettes/ash)
- https://universe.roboflow.com/objectdetection-rrcdv/smoking-detection-dataset -> YOLOv8 v5 (2756 gorsel) -> `..\datasets\smoking-detection-dataset`
- Sinif: 0=ash 1=cigarettes (ash dusurulur).

### 3. Roboflow - cigarette smokers (7758 gorsel, v1) - ONCELIKLI
- https://universe.roboflow.com/smoking-pwchg/cigarette-smokers/dataset/1 -> YOLOv8 -> `..\datasets\cigarette-smokers`
- Sinif listesini indirmeden sonra `README.roboflow.txt`'den dogrula (cigarette/Person/Smoke/Vape olabilir).

### 4. Kaggle - Smoker Detection (siniflandirma, 1120 gorsel 250x250; smoking/notsmoking)
- https://www.kaggle.com/datasets/sujaykapadnis/smoking
- `kaggle datasets download -d sujaykapadnis/smoking`
- Kullanim: agiz-crop siniflandirici icin hard-negatif havuzu (icme, inhaler, telefon).

Not: Sınıf sıraları (class_map) her Roboflow kaynaginda indirme sırasına gore degisebilir;
`training/prepare_dataset.py` calistirmadan once `data.yaml`/`README.roboflow.txt`'i kontrol edip
`smoking_sources.json` ve `weapon_sources.json` icindeki `class_map`'i guncelle.
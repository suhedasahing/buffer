# Eg format paketi (Colab T4)

Prototip icin ham hale getirilmis veri -> tek egni'te -> `best.pt` -> `models/` klasorune kopyala
(yeni isimlerle, eski .pt'lere DOKUNMA).

## Adim 0 - gereksinimler
```
pip install ultralytics opencv-python kaggle
kaggle config  # API key gerekli (kaggle.com/settings -> API)
```

## Adim 1 - veri indir (bkz. download_links.md)
Indirilen zip'ler `../datasets/` altina ac (repo disinda tutmak onerilir; OneDrive yavaslar).

## Adim 2 - harmanla
```
python training/prepare_dataset.py --config training/weapon_sources.json --out training/data/weapon_v2 --class-name weapon
python training/prepare_dataset.py --config training/smoking_sources.json --out training/data/cigarette_v2 --class-name cigarette
```
Cikti: `data.yaml` + `report.json` (kaynak bazinda sayi/sinif/box boyutu dagilimi).

## Adim 3 - silah egitimi (Colab'de)
```
python training/train_yolo.py --data training/data/weapon_v2/data.yaml --model yolov8n --epochs 80 --imgsz 640 --batch auto --device 0 --name weapon_v2
```
- model buyut (yolov8s/m) ancak CPU'da kosacaksa n; gecikme hedefi silah <1 sn.
- Cikti: `runs/weapon_v2/weights/best.pt`. Kesin ismi `models/weapon_v2.pt` ile kopyala.

## Adim 4 - sigara (agiz-seviyesi, iki katmanli)
1. `cigarette_v2` setini egit (Adim 3 gibi, `--class-name cigarette`).
2. Calistirilmada katman: pose ile bas/yuz kutusu -> agiz bolgesi crop -> cigarette kutusu merkez
   agiz bolgesinde mi? (detector.py'ye dokunmadan once bu kurali rule-tabani backend'de test et)
3. Negatif havuz (Kaggle smoking classification) mouth-crop siniflandirici icin.

## Adim 5 - kavga (video)
```
python training/prepare_fight_videos.py --root <airtlab_klonu>/violence-detection-dataset --out training/data/fight_v1
```
- `clips.csv` (clip,label,fps,w,h,frames). Zaman penceresi modeli egitimi icin bu kare setleri + pose
  ciktisi kullanilacak (ayri adim: pose cikar -> dizi -> LSTN/ST-GCN benzeri klasor).
- Kavga hedefi: algilama gecikmesi 2-3 sn.

## Kural
- Her egitim/degisiklik adapterinde eski model silinmez; yeni model ayri adla kaydedilir.
- Calismadaki kod degisiklikleri (detector.py/config.py) ANCAK senin onayinla, olcumle yapilir.
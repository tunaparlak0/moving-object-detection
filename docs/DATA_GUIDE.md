# Veri Toplama ve Etiketleme Rehberi

## 1. Gereksinimler (proje tanımından)
| Kriter | Değer |
|---|---|
| Video sayısı | **≥ 25 orijinal video** |
| Çözünürlük | **≥ 512x512** (genişlik VE yükseklik ≥ 512) |
| Süre | **≥ 15 sn** |
| Ground truth | Yalnızca **gerçekten hareket eden** nesneler için bounding box |

> "Orijinal" = bu problem için zaten hazırlanmış/etiketlenmiş bir veri setinden **alınmamış** video
> (hocanın açıklaması). Pexels/Pixabay gibi stok videolar kabul. Hazır veri setleri (VIVID, CDnet PTZ,
> VisDrone) yalnızca geliştirme/kıyaslama için kullanılabilir, 25 videoya sayılmaz.
>
> **Algoritma sınıftan bağımsız olmalı:** araç/insan *tanıma* yok; hareket eden **her şey** bulunmalı.
> Bu yüzden GT'de de gerçekten hareket eden **tüm** nesneler olmalı (sadece araçlar değil). Sınıf adı
> (vehicle / two_wheeler / person) yalnızca analiz için ek bilgidir.

Kaynak: Pexels/Pixabay drone videoları (kaynak linkleri `data/manifest.csv` içinde). Her video
**15–16 sn**'ye kırpılır; numaralar kesintisiz `vid_001..vid_025`. Çıkarılan bir videonun yerine
gelen yeni video aynı numarayı alır (diğer videoların numarası ve etiketleri değişmez).

**Etiketlenmiş bir videonun dosyası bir daha değiştirilmez** (kırpma / fps değişimi kare
numaralarını kaydırır ve etiketleri bozar).

## 2. Dosya düzeni ve isimlendirme
```
data/videos/vid_001.mp4 ... vid_025.mp4
data/annotations/vid_001.txt ...         (MOT formatı, aşağıda)
data/manifest.csv                        (her videonun meta bilgisi)
```
`manifest.csv` sütunları: `video_id,platform,camera_motion,scene,notes`

## 3. Etiketleme — CVAT
Araç: **CVAT** (https://app.cvat.ai — ücretsiz hesap, ya da yerel Docker kurulumu).

1. Yeni proje → etiket listesi: `vehicle` (araba/kamyon/otobüs), `two_wheeler` (bisiklet/motosiklet), `person`.
   Bu üçüne uymayan hareketli nesneler (hayvan, tekne, el arabası…) de etiketlenir; en yakın sınıfı seçin.
2. Her video için bir task oluşturun, videoyu yükleyin.
3. **Track modu** kullanın (Shape değil): nesneye kutu çizip birkaç karede bir anahtar
   kare (keyframe) ayarlayın; CVAT aradaki kareleri enterpolasyonla doldurur.
4. Dışa aktarma: **Export → "MOT 1.1"** → `python -m src.import_cvat` (bkz. 3b).

### Etiketleme kuralları (en önemli kısım)
- **Sadece o an gerçekten hareket eden** nesneleri etiketleyin.
- Park halindeki araç, duran insan, ağaç vb. **etiketlenmez** — kamera hareketiyle
  görüntüde kaysa bile.
- Nesne durursa: durduğu kareden itibaren track'i **"outside"** yapın; tekrar hareket
  ederse yeniden başlatın (aynı ID olabilir).
- Kutu nesneyi sıkıca sarmalı; %50'den fazla kapanmış (occluded) nesneyi CVAT'ta
  "occluded" işaretleyin.
- Çok küçük nesneler (< 10x10 piksel) etiketlenmez.
- Kare dışına çıkan nesne: görünür kısmı kutulanır, tamamen çıkınca track biter.

## 3b. Yarı otomatik ön etiketleme (önerilen)
```powershell
.venv\Scripts\python -m src.preannotate vid_002      # veya: all
```
- Dedektör (UAVDT YOLOv9s + COCO YOLO11s) → ego-motion telafili takip → hareket filtresi.
- Çıktılar `results/preannot/`: `vid_XXX_cvat.zip` (**CVAT'a yüklenecek**), `vid_XXX.txt`
  (taslağın kendisi, `--drafts` değerlendirmesi için). Önizleme videosu için `--preview`.
- CVAT: task'ı oluştur (Frame step = 1, ayarlara dokunma) → **Actions → Upload annotations →
  "CVAT 1.1"** → `vid_XXX_cvat.zip` → Replace.
- Zip'te kutular yalnızca **~0,2 sn'de bir anahtar karede** bulunur (`--key-s`, 0 = her kare);
  CVAT aradaki kareleri enterpolasyonla doldurur. Sadece anahtar kareleri düzeltin, ara kareleri
  kontrol için oynatın. 0,5 sn'den uzun kesintide iz "outside" olur, sonra aynı ID ile döner.
  (CVAT'a **MOT formatında yükleme yapmayın:** MOT'ta eksik kareyi "nesne yok" sayar.)
- **Sadece her 15 karede bir kontrol edilir:** CVAT **Settings → Player → Player step = 15**;
  0. karede başla, düzelt, **»** ile 15 kare atla (0, 15, 30, …), videonun sonuna kadar.
  Her durakta o kare **hatasız** olmalı: hareket eden her nesne (≥ 10x10 px) sıkı kutuda,
  duran/park halindeki hiçbir nesnede kutu yok (kaldırmak için o karede **Outside**; silme
  tüm izi siler). Aradaki karelere bakmak gerekmez — değerlendirme (`src.evaluate`, varsayılan
  `--frame-step 15`) yalnızca 0, 15, 30, … karelerini sayar.
- Dışa aktarma her zaman **"MOT 1.1"** ile yapılır (enterpolasyonlu tüm kareler gelir).
  Zip'i (adında `vid_XXX` geçsin) `data/annotations/` içine koyup `python -m src.import_cvat`
  çalıştırın: `data/annotations/vid_XXX.txt` yazılır, sınıf sırası isimden eşlenir, zip
  `data/cvat_exports/` altına taşınır.
- **Taslak sadece dedektörün bildiği sınıfları içerir** — hareket eden başka bir şey (hayvan vb.)
  varsa elle ekleyin. YOLO yalnızca etiketlemeyi hızlandırmak içindir, tespit algoritmasının parçası değildir.
- Sonra elle düzelt: kaçırılan hareketli nesneleri ekle, yanlış tutulan statik olanları sil,
  kayan kutuları düzelt, ID değişimlerini birleştir. Ardından "MOT 1.1" olarak dışa aktar.
- Filtre eşiğini değiştirmek için (tespit önbellekten okunur, hızlıdır):
  `python -m src.preannotate vid_002 --speed-thresh 0.4`

> Ön etiketler TASLAKTIR; değerlendirmede kullanılan GT mutlaka elle kontrol edilmiş olmalı.

## 4. MOT ground truth formatı
Her satır bir karedeki bir nesne:
```
frame, id, x, y, w, h, not_ignored, class, visibility
```
- `frame` 1'den başlar, `x,y` sol üst köşe (piksel), `w,h` genişlik/yükseklik.

## 5. Doğrulama
```powershell
.venv\Scripts\python -m src.check_videos            # gereksinim kontrolü
.venv\Scripts\python -m src.visualize_gt vid_001   # GT kutularını video üzerinde göster
```

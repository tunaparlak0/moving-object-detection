# Veri Toplama ve Etiketleme Rehberi

## 1. Gereksinimler (proje tanımından)
| Kriter | Değer |
|---|---|
| Video sayısı | **≥ 25 orijinal video** |
| Çözünürlük | **≥ 512x512** (genişlik VE yükseklik ≥ 512) |
| Süre | **≥ 15 sn** |
| Ground truth | Yalnızca **gerçekten hareket eden** nesneler için bounding box |

> "Orijinal" = sizin çektiğiniz video. Hazır veri setleri (VIVID, CDnet PTZ, VisDrone) yalnızca
> geliştirme/kıyaslama için kullanılabilir, 25 videoya sayılmamalı.

Hedef: **~30 video** çekin (bozuk/elenen olursa 25 altına düşmeyin).

## 2. Çekim planı — çeşitlilik matrisi
Kamera hareketi türü ve sahne çeşitliliği, ego-motion kompanzasyonunu farklı zorluklarda test eder.

| Platform | Kamera hareketi | Örnek sahne | Hedef adet |
|---|---|---|---|
| Elde (telefon) | Yavaş pan / tilt | Kaldırımda yürüyen insanlar, park | 6 |
| Elde (telefon) | Yürürken çekim (titreşimli) | Kampüs, cadde | 5 |
| Araç içi (dashcam / telefon tutucu) | İleri hareket | Trafik, yayalar, bisikletler | 6 |
| Yüksek nokta (balkon/köprü) | Pan + zoom yok | Otopark, kavşak (gözetim kamerası benzeri) | 5 |
| UAV / drone (varsa) | Uçuş, yörünge | Yol, meydan | 5 |
| Zor durumlar | Hızlı pan, gölge, çok nesne, durup kalkan nesne | Karışık | 3 |

### Çekim ipuçları
- Telefon ayarı: **1080p (1920x1080) veya 720p, 30 fps**, **yatay** tutun.
- **20–30 sn** çekin (15 sn sınırının rahat üstünde kalsın; sonra kırpılabilir).
- Dijital zoom ve otomatik video sabitlemeyi (stabilization) mümkünse **kapatın** —
  sabitleme ego-motion'ı kısmen gizler, proje amacına ters düşer.
- Her videoda **hem hareketli hem statik** nesneler olsun (park etmiş araçlar, direkler, ağaçlar):
  bunlar kamera hareketi yüzünden "hareket ediyormuş gibi" görünecek ve sistemin bunları
  elemesi beklenir.
- Hareketli nesne sayısı: video başına 1–10 arası ideal (etiketleme yükü için).
- Yüzler/plakalar: kamusal alan çekimlerinde gerekirse raporda bulanıklaştırın.

## 3. Dosya düzeni ve isimlendirme
```
data/videos/vid_001.mp4 ... vid_030.mp4
data/annotations/vid_001.txt ...         (MOT formatı, aşağıda)
data/manifest.csv                        (her videonun meta bilgisi)
```
`manifest.csv` sütunları: `video_id,platform,camera_motion,scene,notes`

## 4. Etiketleme — CVAT
Araç: **CVAT** (https://app.cvat.ai — ücretsiz hesap, ya da yerel Docker kurulumu).

1. Yeni proje → tek etiket listesi: `person`, `car`, `bicycle`, `motorbike`, `other`.
2. Her video için bir task oluşturun, videoyu yükleyin.
3. **Track modu** kullanın (Shape değil): nesneye kutu çizip birkaç karede bir anahtar
   kare (keyframe) ayarlayın; CVAT aradaki kareleri enterpolasyonla doldurur.
4. Dışa aktarma: **Export → "MOT 1.1"** → zip içindeki `gt/gt.txt` dosyasını
   `data/annotations/vid_XXX.txt` olarak kaydedin.

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

## 5. MOT ground truth formatı
Her satır bir karedeki bir nesne:
```
frame, id, x, y, w, h, not_ignored, class, visibility
```
- `frame` 1'den başlar, `x,y` sol üst köşe (piksel), `w,h` genişlik/yükseklik.

## 6. Doğrulama
```powershell
.venv\Scripts\python -m src.check_videos            # gereksinim kontrolü
.venv\Scripts\python -m src.visualize_gt vid_001   # GT kutularını video üzerinde göster
```

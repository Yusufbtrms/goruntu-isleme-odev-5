# Ödev 5 Raporu: Histogram, Gri Dönüşüm, Renk Uzayları, X-ray Renklendirme ve CUDA

Bu rapordaki sayılar, programların ekran çıktılarından alınmıştır. Kullanılan programlar ve çıktıları:

| Görev | Program | Çıktı dosyaları |
|---|---|---|
| 1, 2, 3 | `odev5_1_3_histogram_gri.cpp` | `histogram_gri_sonuclari.txt`, `cikti_histogram_gri/` |
| 4 | `odev5_4_renk_uzaylari.cpp` | `renk_uzaylari_sonuclari.txt`, `cikti_renk_uzaylari/` |
| 5 | `odev5_5_xray_renklendirme.cpp` | `xray_sonuclari.txt`, `cikti_xray/` |
| Haftalık (CUDA) | `odev5_cuda.cu` | `cuda_sonuclari.txt`, `cuda_sonuclari_2.txt`, `cikti_cuda/` |

**Ortam.** CPU programları: Windows, MSYS2 UCRT64, g++ -O2, OpenCV 5.1.0-pre. CUDA programı: Google Colab, Tesla T4 (hesaplama yeteneği 7.5, 40 işlemci), OpenCV 4.6.0, `nvcc -O2 --fmad=false` (çarp-topla kaynaştırması kapalı, GPU ve CPU'nun aynı ondalıklı işlem sırasını izlemesi için). Görev 1-3 için kullanılan renkli resim `Cat.jpg` (1600x1598, 2 556 800 piksel); Görev 4 aynı resmin en uzun kenarı 800'e küçültülmüş haliyle (800x799, 639 200 piksel, `INTER_AREA`) çalıştırıldı. OpenCV resmi B, G, R sırasıyla okur; tüm formüller bu sıraya göre uygulandı.

---

## 1. Görev 1 ve 2: Kanal histogramları ve kendi `calcHist` fonksiyonumuz

OpenCV ile okunan renkli resim `cv::split` ile B, G, R kanallarına ayrıldı. Her kanal için 256 kutulu histogram, önce OpenCV'nin `cv::calcHist` fonksiyonuyla, sonra kendi yazdığımız `myCalcHist` ile çıkarıldı. Kendi fonksiyonumuz her pikselin değerinden kutu numarasını (`değer x kutu sayısı / 256`, tam sayı bölme) hesaplayıp ilgili kutuyu artırır.

| Kanal | Toplam piksel | Ortalama | En yüksek kutu | En yüksek kutudaki piksel | Kendi - OpenCV (mutlak fark) |
|---|---|---|---|---|---|
| Mavi (B) | 2 556 800 | 120.74 | 202 | 34 709 | 0 |
| Yeşil (G) | 2 556 800 | 148.83 | 225 | 37 060 | 0 |
| Kırmızı (R) | 2 556 800 | 175.92 | 226 | 43 243 | 0 |

Kendi fonksiyonumuz, OpenCV'yle 256, 128, 64, 16 ve 10 kutu için de birebir aynı sonucu verdi (üç kanalın toplam mutlak farkı hepsinde 0). Kanal ortalamaları R > G > B sırasındadır: resim sıcak tonludur (kırmızı baskın). Histogram grafikleri `1_histogramlar_opencv.png` ve `2_histogramlar_kendi.png` dosyalarındadır.

---

## 2. Görev 3: Gri tonlu dönüşüm ve hata değeri

RGB resim, hem OpenCV'nin `cvtColor(BGR2GRAY)` fonksiyonuyla hem de kendi fonksiyonlarımızla tek kanallı gri resme çevrildi. Hata değeri, eşleşen piksellerin mutlak farklarının toplamıdır. "Doğru" dönüşüm, ağırlıkların kendi kanallarına uygulanmasıdır: gri = 0.299 R + 0.587 G + 0.114 B. OpenCV bellekte B, G, R sırasıyla tuttuğu için en sık yapılan hata, kırmızı ağırlığını mavi kanala uygulamaktır; bu hatanın büyüklüğü de ölçüldü.

| Fonksiyon | Mutlak fark toplamı | Farklı piksel | Oran % | En büyük fark |
|---|---|---|---|---|
| A) Kendi, kayan nokta (0.299 R + 0.587 G + 0.114 B, yuvarlamalı) | 0 | 0 | 0.000 | 0 |
| B) Kendi, 15 bit sabit nokta ((9798 R + 19235 G + 3735 B + 16384) >> 15) | 0 | 0 | 0.000 | 0 |
| E) Kendi, 14 bit sabit nokta ((4899 R + 9617 G + 1868 B + 8192) >> 14) | 0 | 0 | 0.000 | 0 |
| C) Basit ortalama (R + G + B) / 3 | 13 799 193 | 2 536 186 | 99.194 | 17 |
| D) Yanlış kanal sırası (B'ye R ağırlığı) | 26 101 361 | 2 554 641 | 99.916 | 29 |

(Toplam 2 556 800 piksel.)

**Hangi sabit nokta aritmetiğini OpenCV kullanıyor?** Orijinal resimde A, B ve E üçü de sıfır hata verdi, yani bu resim üçünü birbirinden ayıramıyor. Küçültülmüş resimde (800x799, Görev 4 programı) ayrım ortaya çıktı:

| Gri fonksiyonu (800x799 resim, OpenCV 5.1.0-pre'ye göre) | Mutlak fark toplamı | Farklı piksel | Oran % |
|---|---|---|---|
| 14 bit sabit nokta | 602 | 602 | 0.094 |
| **15 bit sabit nokta** | **0** | **0** | **0.000** |
| float32 (0.299f, 0.587f, 0.114f) | 493 | 493 | 0.077 |
| double (0.299, 0.587, 0.114) | 571 | 571 | 0.089 |

Dört fonksiyondan yalnızca 15 bit sabit nokta OpenCV 5.1.0-pre ile sıfır hata verdi; diğerleri bir pikselde 1 seviye farklı çıktı. Bu bulgu yalnızca bu OpenCV sürümü ve bu iki resim için geçerlidir.

**C ve D'nin hata büyüklüğü.** D için piksel başına ortalama hata 26 101 361 / 2 556 800 = 10.2'dir. Kanal ortalamalarıyla hesap: ağırlık farkı (0.299 - 0.114) = 0.185, R - B ortalaması 175.92 - 120.74 = 55.18, çarpım yaklaşık 10.2, ölçümle uyumlu (resimde kırmızı maviden çok baskın olduğu için). C için ortalama hata 13 799 193 / 2 556 800 = 5.40; işaretli ortalama farkın kanal ortalamalarından hesabı (0.0343 R - 0.2537 G + 0.2193 B) yaklaşık -5.25, uyumlu. PDF'in belirttiği gibi basit ortalama algısal olarak daha zayıftır; ölçülen fark, resmin %99'unda gri değerinin değiştiğini gösterir.

---

## 3. Görev 4: Renk uzayı dönüşümleri

### 3.1 Yöntem

PDF'teki renk uzayları için dönüşüm fonksiyonları yazıldı: BGR <-> YCrCb, BGR <-> HSV, BGR <-> CIELAB, BGR <-> HSI, BGR -> normalize rgb, BGR -> gri. Fonksiyonlar C++'ta **fonksiyon göstericileriyle** kullanıldı: `using ConvFn = cv::Mat (*)(const cv::Mat&)` türünde göstericiler tablolarda tutuldu ve dönüşümler göstericiler üzerinden çağrıldı (ileri dönüşümler, ters dönüşümler ve BGR merkezli uzaydan uzaya tablosu).

Her dönüşüm için iki tür fonksiyon yazıldı:

- **PDF formülü:** Ders kitabı formülünün çift duyarlıklı uygulaması, sonuç OpenCV'nin 8-bit gösterimine (H = derece / 2, Lab için L x 255 / 100, a ve b + 128) yuvarlanarak çevrilir.
- **OpenCV birebir:** OpenCV'nin 8-bit hesabını (sabit nokta tam sayı aritmetiği) taklit eden sürüm. YCrCb (14 bit, katsayılar 4899, 9617, 1868, 11682, 9241), HSV (12 bit bölme tabloları) ve gri için yazıldı. Lab için OpenCV arama tabloları kullandığından birebir taklit yazılmadı.

OpenCV'de HSI ve normalize rgb için hazır fonksiyon yoktur: HSI bilinen renklerle ve geri dönüşümle, normalize rgb OpenCV matris işlemleriyle kurulan bir referansla doğrulandı.

### 3.2 İleri dönüşümler (BGR -> X): kendi fonksiyon ile OpenCV `cvtColor` farkı

| Dönüşüm | Mutlak fark toplamı | Farklı piksel | Oran % | En büyük | Kanal bazında mutlak fark |
|---|---|---|---|---|---|
| Gri, 15 bit sabit nokta | 0 | 0 | 0.000 | 0 | 0 |
| YCrCb, PDF formülü | 89 937 | 82 331 | 12.880 | 1 | Y 219, Cr 53 237, Cb 36 481 |
| **YCrCb, OpenCV birebir** | **0** | 0 | 0.000 | 0 | 0, 0, 0 |
| HSV, PDF formülü | 219 824 | 11 968 | 1.872 | 180 | H 213 869, S 5 955, V 0 |
| **HSV, OpenCV birebir** | **0** | 0 | 0.000 | 0 | 0, 0, 0 |
| Lab, PDF formülü | 186 274 | 168 532 | 26.366 | 2 | L 89 402, a 57 030, b 39 842 |
| Normalize rgb (OpenCV matris işlemleri referansı) | 0.000000 | 0 | 0.000 | 0.000000 | 0, 0, 0 |

(639 200 piksel.)

**Yorum.**

- OpenCV'yi birebir taklit eden fonksiyonlar (gri 15 bit, YCrCb, HSV) ödevin beklediği sıfır hatayı verdi.
- PDF formüllerinin uygulaması sıfır vermedi: YCrCb'de %12.88 pikselde 1 seviye fark. Olası neden: OpenCV'nin Cr ve Cb katsayıları (0.713 ve 0.564 gibi) PDF'teki katsayılardan (0.5 / 0.701 ve 0.5 / 0.886 gibi) küçük farklıdır ve OpenCV tam sayı yuvarlaması kullanır. Aynı formül ailesi olsa da sonuçlar 8-bit düzeyinde birebir eşleşmez.
- HSV'de PDF formülünün hatası H kanalında yoğunlaşıyor (213 869 / 180 ≈ 1 188, yani hatanın neredeyse tamamı en büyük farkı veren yaklaşık 1 190 pikselden geliyor). Olası neden: 359° civarındaki tonlarda H / 2 değeri 180'e yuvarlanıyor, OpenCV ise 0 veriyor (8-bit H aralığı 0-179; OpenCV'nin tam sayı hesabı 180 üretmez). Bu çıkarım doğrudan pikseller incelenerek doğrulanmadı.
- Lab'de PDF formülü %26.4 pikselde en fazla 2 seviye farklıdır. Olası nedenler: OpenCV Lab'i arama tabloları ve sabit nokta aritmetiğiyle hesaplar; PDF'teki D65 matrisi (0.4124, 0.3576, ...) ve beyaz nokta, OpenCV'nin kullandığı değerlerden küçük farklıdır. İkisi ayrıştırılmadı.

### 3.3 Ters dönüşümler (X -> BGR): aynı girdi (OpenCV'nin ileri dönüşümü), OpenCV'nin ters dönüşümüne karşı

| Dönüşüm | Mutlak fark toplamı | Farklı piksel | Oran % | En büyük | Kanal bazında (B, G, R) |
|---|---|---|---|---|---|
| YCrCb -> BGR, PDF formülü | 37 527 | 37 434 | 5.856 | 1 | 33 353, 105, 4 069 |
| **YCrCb -> BGR, OpenCV birebir** | **0** | 0 | 0.000 | 0 | 0, 0, 0 |
| HSV -> BGR, ders kitabı formülü | 621 224 | 459 267 | 71.850 | 1 | 315 959, 304 368, 897 |
| HSV -> BGR, OpenCV'nin float hesabı, yuvarlamalı | 621 396 | 459 423 | 71.875 | 1 | 315 959, 304 539, 898 |
| **HSV -> BGR, aynı hesap, kesmeli** | **73** | 73 | 0.011 | 1 | 0, 72, 1 |
| Lab -> BGR, PDF formülü | 69 064 | 66 039 | 10.332 | 1 | 32 448, 14 566, 22 050 |

**HSV -> BGR hakkında.** İlk iki satırda B ve G kanallarında yaklaşık %50 pikselde 1 seviye fark vardı, R'de neredeyse hiç yoktu. Bu desen, yuvarlama ile kesme arasındaki farkın tipik izidir (resim kırmızı-turuncu baskın olduğu için R çoğunlukla tam sayı olan V'ye eşit, B ve G kesirli çıkıyor). Hipotez: OpenCV bu sürümde float'tan 8-bit'e çevirirken yuvarlamıyor, kesiyor. Kesmeli varyant bunu doğruladı: hata 621 396'dan 73'e düştü (639 200 pikselde 73 piksel, en fazla 1 seviye). Kalan 73 pikselin sebebini bilmiyoruz; olası bir sebep, OpenCV'nin float işlem sırasının farklı olması (kesme, tam sayıya çok yakın değerlerde küçük float farklarına duyarlıdır).

### 3.4 Gidiş-dönüş (BGR -> X -> BGR), orijinal resme göre hata

8-bit dönüşümler kayıplıdır: ara uzaydaki değerler 8 bite nicemlendiği için geri dönüşüm orijinali birebir vermez.

| Dönüşüm | Mutlak fark toplamı | Farklı piksel | Oran % | En büyük | Kanal bazında (B, G, R) |
|---|---|---|---|---|---|
| YCrCb, kendi (OpenCV birebir) | 346 501 | 252 051 | 39.432 | 1 | 147 340, 126 445, 72 716 |
| YCrCb, OpenCV | 346 501 | 252 051 | 39.432 | 1 | 147 340, 126 445, 72 716 |
| HSV, kendi (OpenCV birebir ileri, kesmeli ters) | 682 793 | 463 267 | 72.476 | 3 | 318 535, 363 372, 886 |
| HSV, OpenCV | 682 808 | 463 264 | 72.476 | 3 | 318 535, 363 388, 885 |
| Lab, kendi (PDF formülü) | 748 773 | 515 227 | 80.605 | 2 | 325 369, 115 874, 307 530 |
| Lab, OpenCV | 802 943 | 524 805 | 82.103 | 4 | 334 824, 154 719, 313 400 |
| HSI, kendi (float; OpenCV'de yok) | 0 | 0 | 0.000 | 0 | 0, 0, 0 |

YCrCb'de kendi ve OpenCV sonuçları aynıdır (fonksiyonlarımız OpenCV'yle birebir olduğu için). HSV'nin gidiş-dönüş hatası (%72.5 pikselde en fazla 3 seviye) büyüktür; H kanalı 2 derecelik adımlarla nicemlendiği için olası bir neden budur, ayrıştırılmadı. Lab'de her iki uygulamanın da hatası büyüktür (%80-82 piksel). HSI float olarak saklandığı için kayıpsızdır.

### 3.5 Uzaydan uzaya dönüşüm (BGR merkezli, fonksiyon göstericili tablo)

Kaynak uzaydaki girdi OpenCV ile üretildi; kendi sonucumuz (önce BGR'ye, sonra hedef uzaya, tablodaki göstericilerle) OpenCV'nin iki adımlı dönüşümüyle karşılaştırıldı:

| Dönüşüm | Mutlak fark toplamı | Farklı piksel | Oran % | En büyük | Kanal bazında |
|---|---|---|---|---|---|
| YCrCb -> HSV | 0 | 0 | 0.000 | 0 | 0, 0, 0 |
| HSV -> YCrCb | 326 | 258 | 0.040 | 1 | 98, 213, 15 |
| YCrCb -> Lab | 147 078 | 136 979 | 21.430 | 1 | 60 336, 52 304, 34 438 |
| Lab -> HSV | 84 012 | 46 961 | 7.347 | 3 | 20 920, 50 296, 12 796 |

HSV ve YCrCb'yi içeren dönüşümler (OpenCV birebir fonksiyonlarla) sıfıra çok yakındır (HSV -> YCrCb'deki 258 piksel, HSV -> BGR'deki kalan 73 pikselin yansımasıdır); Lab'i içerenler PDF formülünün hatasını taşır.

### 3.6 HSI ve normalize rgb: bilinen renklerle doğrulama

| Renk | HSI (hesaplanan) | Beklenen | Normalize rgb (hesaplanan) |
|---|---|---|---|
| Kırmızı | H = 0, S = 1, I = 0.333 | H = 0, S = 1, I = 0.333 | (1, 0, 0) |
| Yeşil | H = 120, S = 1, I = 0.333 | H = 120, S = 1, I = 0.333 | (0, 1, 0) |
| Mavi | H = 240, S = 1, I = 0.333 | H = 240, S = 1, I = 0.333 | (0, 0, 1) |
| Sarı | H = 60, S = 1, I = 0.667 | H = 60, S = 1, I = 0.667 | (0.5, 0.5, 0) |
| Beyaz | H = 0 (tanımsız), S = 0, I = 1 | H tanımsız, S = 0, I = 1 | (1/3, 1/3, 1/3) |
| Gri 128 | H = 0 (tanımsız), S = 0, I = 0.502 | H tanımsız, S = 0, I = 0.502 | (1/3, 1/3, 1/3) |
| Siyah | H = 0, S = 0 (tanımsız), I = 0 | H, S tanımsız, I = 0 | (0, 0, 0) (tanımsız) |

Tanımsız durumlar (PDF'in belirttiği gibi gri piksellerde H, siyahta S ve normalize rgb) 0 olarak alındı.

---

## 4. Görev 5: X-ray görüntülerini gri yapıp renklendirme

### 4.1 Veri

Kaggle HUMS X-ray veri setinin `ThreadsRGB` klasöründen 5 görüntü kullanıldı: `image_0_264.jpg` (825x573), `image_0_287.jpg` (582x483), `image_0_294.jpg` (654x520), `image_0_380.jpg` (722x507), `image_0_387.jpg` (778x503). Görüntüler, tarayıcının malzeme renk kodlamasını taşıyor: turuncu (organik malzeme), yeşil (karışık), mavi (inorganik, metal), koyu mavi (kalın metal), arka plan krem/soluk sarı. Bu, iki enerjili (dual-energy) tarayıcıların standart kodlamasıdır: **ton malzeme türünü, parlaklık yoğunluğu taşır.**

### 4.2 Yöntem

1. Her görüntü kendi 15 bit sabit nokta fonksiyonumuzla gri tonlu tek kanallı resme çevrildi.
2. Gri resim şu renklendirmelerle boyandı: (a) kendi paletimiz, (b) OpenCV JET, TURBO, VIRIDIS paletleri, (c) bu üç paletin gri düzeyin tersine (255 - g) uygulanmış halleri, (d) "üst sınır" (oracle): orijinal renkli resimden her gri düzey için o düzeydeki piksellerin ortalama Lab rengi.
3. Renklendirmeler orijinal renkli resimle karşılaştırıldı.

**Kendi paletimiz.** Gri düzey g için renk, YCbCr uzayında Y = g alınarak, (Cb, Cr) değerleri beş çapa renk arasında doğrusal geçişle belirlenerek üretildi. Çapa renkler (artan parlaklık sırasıyla): koyu mavi (15, 45, 160), mavi (30, 125, 235), yeşil (130, 165, 30), turuncu (240, 140, 30), soluk sarı (255, 247, 190); siyah ve beyaz renksiz uçlardır. Çapaların gri düzeyleri yaklaşık 49, 109, 139, 157 ve 243'tür. Böylece renklendirilmiş resmin gri hali girdi gri resmin aynısı olur (parlaklık sapması 0.00): renk ek bilgi olarak eklenir, yoğunluk bilgisi bozulmaz.

**Ölçütler.**

- **Ortalama ΔE76:** Renklendirilmiş resim ile orijinal arasındaki Lab uzayında ortalama Öklid renk farkı.
- **Sınıf uyumu:** Pikseller HSV'de renk tonuna göre 4 sınıfa ayrıldı: arka plan/soluk (S < 60), turuncu (H < 50° ya da H ≥ 330°), yeşil (50° ≤ H < 150°), mavi (H ≥ 150°). Sınıf uyumu, renklendirmedeki sınıfın orijinaldeki sınıfla aynı olduğu piksellerin oranıdır. (Eşikler seçilmiştir; resimler üzerinde görsel olarak doğrulanmadı.)
- **Parlaklık sapması:** Renklendirilmiş resmin gri hali ile girdi gri resim arasındaki ortalama mutlak fark.

### 4.3 Orijinaldeki sınıfların gri düzeyleri

Orijinalin renk tonuna göre belirlenen sınıfların, gri resimdeki ortalama ± standart sapması:

| Görüntü | Arka plan (%) | Arka plan gri | Turuncu (%) | Turuncu gri | Yeşil (%) | Yeşil gri | Mavi (%) | Mavi gri |
|---|---|---|---|---|---|---|---|---|
| image_0_264 | 59.17 | 247.2 ± 16.1 | 10.89 | 204.4 ± 25.6 | 14.93 | 215.9 ± 46.4 | 15.01 | 100.4 ± 36.5 |
| image_0_287 | 47.47 | 251.5 ± 10.9 | 4.55 | 181.5 ± 30.2 | 16.78 | 208.9 ± 43.3 | 31.20 | 107.9 ± 59.5 |
| image_0_294 | 60.74 | 250.7 ± 12.6 | 10.65 | 197.0 ± 43.7 | 16.24 | 199.7 ± 50.3 | 12.37 | 67.3 ± 43.0 |
| image_0_380 | 37.31 | 242.8 ± 23.4 | 32.02 | 209.9 ± 23.2 | 18.41 | 217.4 ± 42.5 | 12.27 | 103.8 ± 33.4 |
| image_0_387 | 57.76 | 251.3 ± 10.0 | 8.40 | 192.8 ± 24.6 | 13.67 | 212.6 ± 43.7 | 20.17 | 126.0 ± 37.4 |

**Bulgu.** Mavi (metal) sınıfı gri düzeyde belirgin biçimde daha koyudur (67-126). Turuncu (181-210) ve yeşil (200-217) sınıflarının gri ortalamaları birbirine çok yakındır ve standart sapmaları büyüktür (23-50): **gri resimden turuncu ve yeşil sınıflarını güvenilir biçimde ayırmak mümkün değildir.** Bu, paletin çapalarını tasarlarken varsaydığım değerlerden (turuncu 157, yeşil 139) farklıdır: gerçek turuncu ve yeşil pikseller ortalamada daha açıktır.

### 4.4 Palet karşılaştırması

Beş görüntünün ortalaması:

| Palet | Ortalama ΔE76 | Sınıf uyumu % | Parlaklık sapması |
|---|---|---|---|
| **Kendi paletimiz (parlaklık korumalı)** | **19.63** | **76.60** | **0.00** |
| OpenCV JET | 90.22 | 24.04 | 153.48 |
| OpenCV TURBO | 88.44 | 22.36 | 152.69 |
| OpenCV VIRIDIS | 64.43 | 29.62 | 34.62 |
| OpenCV JET (ters gri) | 128.57 | 7.10 | 179.61 |
| OpenCV TURBO (ters gri) | 101.33 | 5.14 | 151.33 |
| OpenCV VIRIDIS (ters gri) | 98.62 | 12.24 | 165.93 |
| Üst sınır (gri düzeye göre en iyi renk) | 12.14 | 79.62 | 1.39 |

Görüntü bazında, kendi paletimiz ve üst sınır:

| Görüntü | Kendi: ΔE76 | Kendi: ΔE > 25 piksel % | Kendi: sınıf uyumu % | Üst sınır: ΔE76 | Üst sınır: ΔE > 25 piksel % | Üst sınır: sınıf uyumu % |
|---|---|---|---|---|---|---|
| image_0_264 | 16.96 | 16.76 | 78.45 | 11.50 | 14.48 | 79.43 |
| image_0_287 | 28.78 | 34.06 | 68.37 | 14.92 | 22.65 | 79.13 |
| image_0_294 | 15.00 | 16.21 | 83.57 | 8.30 | 12.63 | 81.97 |
| image_0_380 | 16.12 | 18.24 | 76.57 | 11.83 | 15.58 | 76.41 |
| image_0_387 | 21.27 | 21.69 | 76.07 | 14.18 | 19.89 | 81.15 |

### 4.5 Sınıf karışıklık matrisi (kendi paletimiz, 5 görüntü toplamı)

Satır: orijinaldeki sınıf, sütun: kendi paletimizle renklendirilen resimdeki sınıf (satır yüzdesi):

| Orijinaldeki sınıf | Arka plan | Turuncu | Yeşil | Mavi | Piksel |
|---|---|---|---|---|---|
| Arka plan | **92.8** | 4.8 | 2.1 | 0.3 | 982 318 |
| Turuncu | 0.0 | **94.1** | 3.1 | 2.8 | 250 542 |
| Yeşil | 29.2 | 38.7 | **23.1** | 8.9 | 293 875 |
| Mavi | 0.0 | 20.5 | 14.9 | **64.6** | 324 564 |

Görüntü bazında köşegen (doğru boyanma) oranları:

| Görüntü | Arka plan | Turuncu | Yeşil | Mavi |
|---|---|---|---|---|
| image_0_264 | 90.5 | 95.1 | 21.1 | 75.9 |
| image_0_287 | 95.3 | 85.2 | 17.2 | 52.5 |
| image_0_294 | 97.0 | 83.0 | 28.0 | 91.1 |
| image_0_380 | 83.6 | 97.1 | 27.0 | 75.9 |
| image_0_387 | 95.9 | 97.7 | 21.3 | 47.6 |

### 4.6 Yorum

**Palet seçiminin gerekçesi.**

1. **Operatör aşinalığı:** Güvenlik operatörleri turuncu = organik, yeşil = karışık, mavi = inorganik/metal kodlamasıyla eğitilir; aynı renk dilini kullanmak yeni bir eğitim gerektirmez.
2. **Parlaklık korunumu:** Palet, her gri düzey için Y = g olacak şekilde kurulduğundan renkli resmin gri hali girdinin aynısıdır (sapma 0.00). Yoğun malzeme koyu, ince malzeme açık kalır; renk, yoğunluk bilgisini bozmadan ek bilgi olarak biner. JET ve TURBO gibi gökkuşağı paletleri parlaklığı tekdüze değiştirmez, parlaklık sapmaları 150'nin üzerindedir; VIRIDIS'in parlaklığı monoton ama girdiden farklıdır (sapma 34.6). Bu paletlerde, veride olmayan sahte kenarlar ve bölgeler algılanabilir (gökkuşağı paletlerinin bilinen bir sorunu; bu raporda ölçülmedi).
3. **Soluk uç:** Paletin yüksek gri düzeyleri (açık, ince malzeme ve arka plan) soluk sarı ve beyaza gider, scanner'ın krem arka planına benzer; piksellerin %37-61'i bu bölgededir.

**Orijinalle fark (sayısal).**

- Kendi paletimizin orijinale uzaklığı ortalama ΔE76 = 19.63; üst sınır 12.14. Yani ΔE'nin yaklaşık 12'si (yaklaşık %62'si) gri düzeyden renk çıkarmanın kaçınılmaz bedelidir (üst sınır, her gri düzeye ortalama rengi vererek bulundu; ortalama rengin ΔE76'yı en aza indirdiği kesin değildir, bu yüzden ΔE için yaklaşık bir alt sınırdır). Paletimiz bunun üstüne yaklaşık 7.5 birim ekliyor.
- Sınıf uyumunda paletimiz 76.60, üst sınır 79.62; ama üst sınır sınıf uyumu için kesin bir sınır değildir (image_0_294 ve image_0_380'de paletimiz üst sınırı geçiyor: 83.57 ve 76.57, üst sınırda 81.97 ve 76.41).

**Farkın nedeni: ton bilgisinin kaybı.** Tarayıcı malzemeyi tonla kodlar; griye çevirince ton kaybolur ve gri düzeyden yalnızca yoğunluk kalır. Gri düzeyde mavi (metal) ayrılabilirken (ortalama 67-126), turuncu ve yeşil iç içedir (181-217, standart sapma 23-50). Karışıklık matrisi bunu doğruluyor: turuncu %94.1 ve arka plan %92.8 doğru boyanıyor, mavi %64.6, ama yeşil yalnızca %23.1: yeşil piksellerin %29.2'si arka plana, %38.7'si turuncuya gidiyor. Gri resimden bu sınıfları ayırmanın bir yolu yoktur; bu, hangi palet seçilirse seçilsin geçerlidir.

**Paletimizin zayıf noktası.** Mavi sınıfın doğru boyanma oranı görüntüye göre çok değişiyor: image_0_294'te %91.1, image_0_387'de %47.6, image_0_287'de %52.5. İki düşük görüntüde mavi piksellerin yaklaşık üçte biri turuncuya (%34.7 ve %31.9), %12.8-20.5'i yeşile gidiyor. Bu iki görüntüde mavi sınıfın gri ortalaması (107.9 ve 126.0) image_0_294'ünkinden (67.3) yüksektir, yani daha açık tonlu metaller var; paletin mavi çapaları (gri 49 ve 109) bu açık maviler için fazla koyu, açık maviler yeşil ve turuncu bölgeye düşüyor. Bu çıkarım sayılara dayanır. Görsel incelemede (aşağıya bakınız) bıçak gövdelerinin çoğunlukla mavi göründüğü, açık mavi metallerin turuncuya boyandığının net doğrulanamadığı görüldü; mavi sınıftaki hata nesne kenarlarındaki geçiş piksellerinden de gelebilir.

**OpenCV paletleri.** JET, TURBO ve VIRIDIS'in skorları kötüdür (ΔE76 64-90, sınıf uyumu %22-30). Gri düzeyi tersine çevirmenin skoru düzelteceği varsayımı yanlış çıktı: ters gri hepsini kötüleştirdi (JET 128.6, TURBO 101.3, VIRIDIS 98.6). Her iki yönde de bu paletlerin soluk bir ucu yoktur; piksellerin %37-61'i olan krem arka plan doygun ya da koyu renklere boyanır. Olası neden budur; doğrudan doğrulanmadı. Bu skorlar scanner'ın renk kodlamasına benzerliği ölçer; paletlerin başka kullanımlardaki kalitesini ölçmez.

**Operatör açısından sonuç.** Gri tek enerjili bir görüntüden renklendirme, malzeme türünü (organik / karışık) geri getiremez; yalnızca yoğunluk algısını vurgular ve metali (koyu) organikten (açık) ayırmaya yardım eder. Malzeme ayrımı için iki enerjili orijinal görüntü kullanılmalıdır.

**Görsel inceleme.** İki karşılaştırma görüntüsü incelendi: `5_goruntu2_karsilastirma.png` (test hedefi) ve eğik bir bavulu gösteren 3. görüntünün karşılaştırması. Görüntüler küçültülmüş halde incelendi, ince ayrıntılar seçilemedi; diğer üç görüntünün çıktısı incelenmedi. Gözlemler:

1. **Kendi paletimiz:** Kalın metal bloklar koyu laciverte, bıçak gövdeleri maviye, turuncu (organik) bloklar turuncuya, arka plan beyaza boyanıyor. Bu bölgelerde sonuç orijinale yakındır; karışıklık matrisindeki yüksek doğru boyanma oranlarıyla (arka plan %92.8, turuncu %94.1, mavi %64.6) tutarlıdır.
2. **Yeşil bölgeler turuncu-bej oluyor:** 2. görüntüde soldaki 9.5, 15.9, 22.2 etiketli zeytin-yeşil bloklar ve 3. görüntüde bavulun yeşil çerçevesi turuncu ya da bej tonlarına boyanıyor. Bu, yeşil → turuncu karışıklığını görsel olarak doğrular.
3. **Açık bölgelerde şeftali/turuncu bir ton:** Özellikle 3. görüntüde bavulun içinde ve nesnelerin çevresinde görülüyor. Orijinalde bu bölgeler soluk sarı/krem. Paletin turuncu ile soluk sarı çapaları arasındaki orta-açık gri düzeyleri turuncuya kayıyor; bunun arka plan → turuncu (%4.8) ve yeşil → arka plan (%29.2) hatalarına katkı yaptığı düşünülüyor, ölçülmedi.
4. **Açık mavi metallerin turuncuya boyandığı çıkarımı doğrulanamadı:** Bıçak gövdeleri görüntülerde çoğunlukla mavi görünüyor. Mavi → turuncu hatası (özellikle 2. görüntüde %34.7) nesne kenarlarındaki geçiş piksellerinden ya da ince bölgelerden de gelebilir; bu ayrıştırılmadı.
5. **OpenCV paletleri:** JET ve TURBO'da arka plan koyu kırmızıya, VIRIDIS'te doygun sarıya boyanıyor; ters gri varyantlarında arka plan koyu mavi ya da mora dönüyor. Bu, düşük skorların nedeninin arka planın doygun ya da koyu renklere boyanması olduğu açıklamasıyla uyumludur. Ters gri paletlerde (koyu arka plan, yoğun malzeme kırmızı ve sarı) nesneler görsel olarak belirgin ve okunaklı görünüyor; bu paletlerin düşük skorları orijinal tarayıcı kodlamasına benzemediklerini ölçer, okunaksız olduklarını değil.
6. **"Üst sınır" görüntüsü soluk ve az doygun:** Belirsiz gri düzeylere ortalama renk verildiği için renk farkı (ΔE) düşük, ama renkler silik görünüyor.

---

## 5. Haftalık ödev: Görev 1-5'in CUDA ile gerçeklenmesi

### 5.1 Yöntem

Piksel işlemleri `__host__ __device__` işlevli yapılar (functor) olarak yazıldı: aynı kod hem CPU'da (tek thread) hem GPU'da çalışır, böylece karşılaştırma yalnızca paralel çalıştırmayı ve sayısal kütüphane farklarını sınar. GPU'da genel bir `mapKernel` şablonu her pikseli bir thread'e dağıtır (grid-stride döngüsü, 256 thread / blok).

- **Histogram:** İki çekirdek. Naif: her piksel için global bellekteki 768 kutuya atomik toplama. Paylaşılan bellek: blok başına özel 768 kutuluk histogram (paylaşılan bellekte atomik toplama), sonda global belleğe birleştirme.
- **Gri:** 15 bit sabit nokta.
- **Renk uzayları:** YCrCb, HSV (OpenCV birebir tam sayı hesabı, bölme tabloları yerine tam sayı bölmeyle), HSV ters (float, kesmeli), Lab ve Lab ters (çift duyarlıklı, PDF formülü), HSI ve HSI ters (çift duyarlıklı), normalize rgb (float).
- **X-ray renklendirme:** Gri ve gri -> renk (256 girişli arama tablosu), ayrıca ikisinin tek çekirdekte birleşik hali (BGR -> renk).

**Ölçüm.** CPU: tek thread, 3 tekrarın ortalaması. GPU çekirdek süresi: 30 art arda çalıştırmanın CUDA olay (event) ile ölçülen ortalaması. GPU toplam süresi: GPU'ya yükleme, çekirdek ve sonucu geri indirme dahil, 10 tekrarın ortalaması. Program iki kez çalıştırıldı (aşağıda "1. / 2." sütunları).

**Doğruluk.** GPU sonucu CPU sonucuyla eleman eleman karşılaştırıldı (float çıktılar için eşik: HSI 1e-3, normalize rgb 1e-6). Tüm işlemlerde farklı eleman sayısı **0** ve en büyük fark **0** çıktı (her iki çalıştırmada). Histogramlarda da GPU ve CPU kutu sayıları aynı (mutlak fark toplamı 0).

### 5.2 Süre sonuçları (Cat.jpg, 1600x1598, 2 556 800 piksel)

Her hücrede "1. çalıştırma / 2. çalıştırma" değerleri verilmiştir.

| İşlem | CPU (ms) | GPU çekirdek (ms) | GPU toplam (ms) | Hızlanma, çekirdek | Hızlanma, toplam |
|---|---|---|---|---|---|
| Histogram, naif (global atomik) | 4.304 / 4.377 | 1.318 / 1.319 | 2.892 / 2.908 | 3.3x / 3.3x | 1.5x / 1.5x |
| Histogram, paylaşılan bellek | 4.304 / 4.377 | 0.154 / 0.159 | 1.707 / 1.922 | 28.0x / 27.5x | 2.5x / 2.3x |
| Gri | 3.507 / 3.547 | 0.063 / 0.064 | 2.263 / 2.268 | 55.3x / 55.4x | 1.5x / 1.6x |
| BGR -> YCrCb | 9.257 / 9.201 | 0.089 / 0.089 | 3.406 / 3.449 | 104.5x / 103.9x | 2.7x / 2.7x |
| YCrCb -> BGR | 9.293 / 10.786 | 0.085 / 0.083 | 3.314 / 3.440 | 109.1x / 129.3x | 2.8x / 3.1x |
| BGR -> HSV | 31.941 / 29.634 | 0.132 / 0.123 | 3.431 / 3.348 | 241.2x / 241.8x | 9.3x / 8.9x |
| HSV -> BGR | 28.914 / 27.723 | 0.109 / 0.102 | 3.308 / 3.404 | 266.4x / 272.2x | 8.7x / 8.1x |
| BGR -> Lab | 453.045 / 458.203 | 14.238 / 15.387 | 13.058 / 13.396 | 31.8x / 29.8x | 34.7x / 34.2x |
| Lab -> BGR | 257.630 / 349.362 | 8.471 / 8.468 | 11.632 / 11.828 | 30.4x / 41.3x | 22.1x / 29.5x |
| BGR -> HSI | 89.329 / 103.448 | 2.693 / 2.695 | 11.002 / 11.081 | 33.2x / 38.4x | 8.1x / 9.3x |
| HSI -> BGR | 139.846 / 125.484 | 1.753 / 1.752 | 10.118 / 9.865 | 79.8x / 71.6x | 13.8x / 12.7x |
| BGR -> normalize rgb | 12.104 / 6.440 | 0.194 / 0.194 | 9.233 / 8.312 | 62.4x / 33.2x | 1.3x / 0.8x |

X-ray görüntüsü (`image_0_287.jpg`, 582x483, 281 106 piksel):

| İşlem | CPU (ms) | GPU çekirdek (ms) | GPU toplam (ms) | Hızlanma, çekirdek | Hızlanma, toplam |
|---|---|---|---|---|---|
| X-ray: gri | 0.763 / 0.401 | 0.005 / 0.004 | 0.744 / 0.339 | 157.6x / 89.7x | 1.0x / 1.2x |
| X-ray: gri -> renk (arama tablosu) | 0.466 / 0.247 | 0.057 / 0.006 | 0.525 / 0.332 | 8.2x / 43.2x | 0.9x / 0.7x |
| X-ray: BGR -> renk (tek çekirdekte) | 0.741 / 0.426 | 0.006 / 0.006 | 0.508 / 0.482 | 116.7x / 67.9x | 1.5x / 0.9x |

Kaydedilen GPU çıktıları: `cikti_cuda/cuda_xray_gri.png` ve `cikti_cuda/cuda_xray_renkli.png`.

Kendi CPU sonuçlarımızı bu makinedeki OpenCV 4.6.0 `cvtColor` sonucuyla da karşılaştırdık (bilgi amaçlı): gri, YCrCb ve HSV'de fark 0; Lab'de (PDF formülü) 7 670 400 değerin 748 757'sinde (yaklaşık %9.8) fark var, toplam mutlak fark 749 586 (çoğu 1 seviye), Görev 4'te bulunan PDF-Lab farkıyla uyumlu.

### 5.3 Yorum

- **Doğruluk:** GPU ve CPU sonuçları, `pow`, `cbrt`, `acos` kullanan Lab ve HSI dahil tüm işlemlerde birebir aynıdır. Bu matematik fonksiyonlarında CPU ve GPU kütüphaneleri son basamakta farklı olabilir, 8-bit yuvarlamada fark çıkabilirdi; bu görüntüde çıkmadı. Bu, her görüntü için bir garanti değildir.
- **Çekirdek hızlanması büyük, toplam hızlanma küçük.** Basit işlemlerde (gri, YCrCb, normalize rgb) çekirdek 55-104 kat hızlıyken toplam hızlanma 0.8-2.7 kattır: sürenin neredeyse tamamı veri aktarımıdır. Toplam süreden çekirdek süresini çıkarınca kalan aktarım maliyeti gri için yaklaşık 2.2 ms (7.67 MB yükleme + 2.56 MB indirme), YCrCb için yaklaşık 3.3 ms (7.67 + 7.67 MB); iki durumda da yaklaşık 4.6 GB/s ediyor (kendi aritmetiğimiz). Uygulamada veri GPU'da kalıp birden çok işlem zincirlenirse kazanç artar.
- **Hesap yükü arttıkça toplam hızlanma yaklaşan çekirdek hızlanmasına yaklaşır:** HSV'de toplam hızlanma 8-9 kat, HSI'da 8-14 kat, Lab'de 22-35 kat. Lab'de çekirdek hızlanması (30-41 kat) ve toplam hızlanma (22-35 kat) birbirine yakındır, çünkü hesap, aktarım süresinin yanında büyüktür.
- **Histogram:** Paylaşılan bellek çekirdeği naiften yaklaşık 8.6 kat hızlıdır (0.154 / 1.318 ms). Olası neden: yalnızca 768 kutu var ve milyonlarca piksel aynı kutulara yazıyor; global bellekte bu atomik toplamalar sıraya giriyor, blok başına özel histogram bunu azaltıyor. (Ödev 6'daki Hough oylamasında iki çekirdek aynı hızdaydı; orada 360 binden fazla hücreye oy yayılıyor. Bu karşılaştırma doğrulanmadı, olası bir açıklama.)
- **Lab ve HSI yavaş:** Lab çekirdeği (8.5-15.4 ms), YCrCb'den (0.09 ms) yaklaşık 100 kat yavaştır. Olası neden: çift duyarlıklı (double) hesap; T4'te çift duyarlıklı işlem hızı tek duyarlıklının yaklaşık 1/32'si kadardır. Float'a geçmek hızlandırırdı ama CPU sonucuyla birebir aynı olmazdı. Bu doğrulanmadı.
- **Lab'de toplam süre çekirdek süresinden küçük çıktı** (BGR -> Lab: 13.058 ve 14.238 ms, 13.396 ve 15.387 ms). Bu iki çalıştırmada da tekrarlandı, yani gürültü değil. Olası neden: çekirdek süresi 30 art arda çalıştırmanın ortalaması; sürekli yük altında GPU saat hızı düşmüş olabilir, toplam süre ölçümünde çekirdekler aktarımlarla ayrıldığı için saat hızı daha yüksek kalmış olabilir. Saat hızı izlenmedi, doğrulanmadı.
- **Küçük görüntüde GPU kazanç sağlamıyor:** X-ray görüntüsünde (281 bin piksel) toplam hızlanma 0.7x-1.5x arasında, yani aktarım maliyeti kazancı yutuyor. Kazanç yalnızca çekirdek süresindedir (43-158 kat).
- **Ölçüm kararlılığı:** Çekirdek süreleri iki çalıştırmada neredeyse aynıdır (örneğin gri 0.063 ve 0.064 ms, HSV 0.132 ve 0.123 ms). İlk çalıştırmada X-ray "gri -> renk (arama tablosu)" çekirdeği 0.057 ms çıkmıştı; ikinci çalıştırmada 0.006 ms, yani ilk değer tek seferlik bir sapmaydı. Hafif işlemlerin CPU süreleri çalıştırmalar arasında 2 katına kadar oynuyor (normalize rgb 12.104 ve 6.440 ms; X-ray gri 0.763 ve 0.401 ms), bu yüzden bu satırlardaki hızlanma oranları kararsızdır; Lab gibi ağır işlemlerin süreleri kararlıdır (453 ve 458 ms).

---

## 6. Sınırlılıklar

- Görev 1-4 tek bir resimle (`Cat.jpg`) yapıldı; Görev 4 küçültülmüş (800x799) haliyle çalıştırıldı, Görev 3 orijinal boyutta (ve ek olarak küçültülmüş resimde, Görev 4 programıyla) ölçüldü. Sonuçlar bu resimlere özgüdür; özellikle gri için 14 ve 15 bit sabit nokta farkı orijinal resimde görünmedi.
- OpenCV'ye göre sıfır hata, OpenCV 5.1.0-pre (MSYS2) için doğrulandı. Colab'daki OpenCV 4.6.0 ile yalnızca gri, YCrCb ve HSV ileri dönüşümleri bilgi amaçlı karşılaştırıldı (hepsi 0 fark); ters dönüşümler ve gri için 14/15 bit ayrımı o sürümde sınanmadı.
- Lab için OpenCV'nin tablo tabanlı hesabını birebir taklit eden bir fonksiyon yazılmadı; Lab karşılaştırmaları PDF formülünün OpenCV'den sapmasını gösterir. HSV -> BGR'de 73 piksellik artık hata açıklanamadı. HSV H kanalının 180'e yuvarlanması ve YCrCb katsayı farkları çıkarımdır, piksel düzeyinde doğrulanmadı.
- Görev 5: Yalnızca 5 görüntü kullanıldı. Sınıf ayrımı eşikleri (S < 60, ton sınırları) seçilmiştir, görsel olarak doğrulanmadı. Palet çapaları, tipik tarayıcı renklerinin parlaklığı varsayılarak seçildi; veriler turuncu ve yeşil sınıfların daha açık, açık mavi metallerin paletin mavi çapalarından daha açık olduğunu gösterdi (yani palet bu 5 görüntüye göre ayarlanmadı ve açık metaller için zayıf). "Üst sınır" ΔE için yaklaşık bir alt sınırdır, sınıf uyumu için kesin sınır değildir. Görsel inceleme yalnızca iki karşılaştırma görüntüsüyle (2. ve 3. görüntü) ve küçültülmüş boyutta yapıldı; diğer üç görüntünün çıktısı incelenmedi.
- CUDA: Tek GPU (Tesla T4), tek CPU thread'iyle karşılaştırma, iki çalıştırma. Veri aktarımı sayfalanabilir (pinned olmayan) bellekle yapıldı; sabitlenmiş (pinned) bellek ve CUDA akışlarıyla aktarım ve hesabı örtüştürmek denenmedi. Lab ve HSI çift duyarlıklı hesapla yapıldı. Lab'deki toplam < çekirdek süresi ve histogram karşılaştırması için verilen açıklamalar doğrulanmadı. ΔE ve sınıf uyumu hesapları (Görev 5'in ölçütleri) GPU'ya taşınmadı; yalnızca gri ve renklendirme GPU'da yapıldı.

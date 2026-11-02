// Ödev 5 - Görev 1, 2 ve 3
//
// Görev 1: Renkli bir görüntüyü OpenCV ile oku; her kanal için ayrı ayrı histogram çıkar (OpenCV'nin cv::calcHist'i).
// Görev 2: calcHist fonksiyonunu kendin yaz, her kanal için ayrı ayrı çalıştır, OpenCV'ninkiyle karşılaştır.
// Görev 3: RGB görüntüyü DOĞRU şekilde gri tonlu tek kanallı görüntüye çevir: hem OpenCV'nin hazır fonksiyonu (cvtColor)
//          hem kendi fonksiyonlarımızla; eşleşen piksellerin mutlak farklarının toplamını hata değeri olarak raporla.
//
// Not (kanal sırası): OpenCV renkli resmi B, G, R sırasıyla okur (bellekte p[0] = mavi, p[1] = yeşil, p[2] = kırmızı).
// "Doğru" gri dönüşüm, ağırlıkların kendi kanallarına uygulanmasıdır: gri = 0.299 R + 0.587 G + 0.114 B.
// Yanlış sıra (kırmızı ağırlığını mavi kanala uygulamak) sık yapılan bir hatadır; hatanın büyüklüğünü de göstereceğiz.
//
// Üç farklı kendi gri fonksiyonu:
//   A) gri_kayan_nokta : round(0.299 R + 0.587 G + 0.114 B)                       (ders kitabı formülü)
//   B) gri_sabit_nokta : (9798 R + 19235 G + 3735 B + 16384) >> 15                (OpenCV 5.x'in 8-bit sabit noktalı hesabı; 14 bit sürümü E satırında)
//   C) gri_ortalama    : (R + G + B) / 3                                           (basit ortalama, algısal olarak zayıf)
//   D) gri_yanlis_sira : B kanalına R ağırlığı, R kanalına B ağırlığı (hata örneği)
//
// Kullanım: ./histogram_gri [cikti_dizini] [resim_yolu=images/Cat.jpg]
// Derleme:  g++ -O2 odev5_1_3_histogram_gri.cpp -o histogram_gri.exe -std=c++17 $(pkg-config --cflags --libs opencv5)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fsys = std::filesystem;

// ---------------------------------------------------------------------------------------------
// Görev 2: kendi calcHist fonksiyonumuz
//   kanal : tek kanallı 8-bit resim
//   bins  : kutu sayısı; değer v, [0, 256) aralığında (v * bins / 256) numaralı kutuya düşer
// ---------------------------------------------------------------------------------------------
static std::vector<long long> myCalcHist(const cv::Mat& channel, int bins = 256) {
    std::vector<long long> hist(bins, 0);
    for (int y = 0; y < channel.rows; ++y) {
        const uchar* row = channel.ptr<uchar>(y);
        for (int x = 0; x < channel.cols; ++x) ++hist[(row[x] * bins) / 256];
    }
    return hist;
}

// Görev 1: OpenCV'nin cv::calcHist'i (tek kanal, aralık [0, 256))
static std::vector<long long> openCvCalcHist(const cv::Mat& channel, int bins = 256) {
    cv::Mat h;
    const int channels[] = {0};
    const int histSize[] = {bins};
    const float range[] = {0.0f, 256.0f};
    const float* ranges[] = {range};
    cv::calcHist(&channel, 1, channels, cv::Mat(), h, 1, histSize, ranges, true, false);
    std::vector<long long> out(bins);
    for (int i = 0; i < bins; ++i) out[i] = static_cast<long long>(std::lround(h.at<float>(i)));
    return out;
}

// Histogramı çubuk grafik olarak çizer (her kutu 2 piksel genişlikte, 256 kutu için 512 piksel)
static cv::Mat plotHist(const std::vector<long long>& h, const cv::Scalar& color, const std::string& title, int bins = 256) {
    const int barW = std::max(1, 512 / bins), W = bins * barW, H = 200, top = 26;
    cv::Mat img(H + top + 20, W + 20, CV_8UC3, cv::Scalar(255, 255, 255));
    const long long mx = *std::max_element(h.begin(), h.end());
    for (int i = 0; i < bins; ++i) {
        const int bh = mx > 0 ? static_cast<int>(std::lround(static_cast<double>(h[i]) / mx * H)) : 0;
        cv::rectangle(img, cv::Point(10 + i * barW, top + H - bh), cv::Point(10 + (i + 1) * barW - 1, top + H), color, cv::FILLED);
    }
    cv::rectangle(img, cv::Rect(10, top, W, H), cv::Scalar(120, 120, 120), 1);
    cv::putText(img, title + "  (en yuksek kutu: " + std::to_string(mx) + " piksel)", cv::Point(10, 18), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
    cv::putText(img, "0", cv::Point(8, top + H + 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
    cv::putText(img, "255", cv::Point(W - 12, top + H + 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
    return img;
}

// ---------------------------------------------------------------------------------------------
// Görev 3: kendi gri dönüşüm fonksiyonlarımız (girdi: OpenCV'nin B, G, R sıralı 8-bit resmi)
// ---------------------------------------------------------------------------------------------
static cv::Mat gri_kayan_nokta(const cv::Mat& bgr) {
    cv::Mat g(bgr.size(), CV_8UC1);
    for (int y = 0; y < bgr.rows; ++y)
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b p = bgr.at<cv::Vec3b>(y, x);  // p[0] = B, p[1] = G, p[2] = R
            g.at<uchar>(y, x) = cv::saturate_cast<uchar>(static_cast<int>(std::lround(0.299 * p[2] + 0.587 * p[1] + 0.114 * p[0])));
        }
    return g;
}

static cv::Mat gri_sabit_nokta(const cv::Mat& bgr) {
    cv::Mat g(bgr.size(), CV_8UC1);
    for (int y = 0; y < bgr.rows; ++y)
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b p = bgr.at<cv::Vec3b>(y, x);
            g.at<uchar>(y, x) = static_cast<uchar>((p[2] * 9798 + p[1] * 19235 + p[0] * 3735 + (1 << 14)) >> 15);  // 9798+19235+3735 = 32768 = 2^15
        }
    return g;
}

// Eski OpenCV sürümlerinin 14 bit sabit noktalı hesabı (bu sürümde bazı piksellerde 1 seviye farklı çıkabilir)
static cv::Mat gri_sabit_nokta14(const cv::Mat& bgr) {
    cv::Mat g(bgr.size(), CV_8UC1);
    for (int y = 0; y < bgr.rows; ++y)
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b p = bgr.at<cv::Vec3b>(y, x);
            g.at<uchar>(y, x) = static_cast<uchar>((p[2] * 4899 + p[1] * 9617 + p[0] * 1868 + (1 << 13)) >> 14);  // 4899+9617+1868 = 16384 = 2^14
        }
    return g;
}

static cv::Mat gri_ortalama(const cv::Mat& bgr) {
    cv::Mat g(bgr.size(), CV_8UC1);
    for (int y = 0; y < bgr.rows; ++y)
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b p = bgr.at<cv::Vec3b>(y, x);
            g.at<uchar>(y, x) = static_cast<uchar>(std::lround((p[0] + p[1] + p[2]) / 3.0));
        }
    return g;
}

static cv::Mat gri_yanlis_sira(const cv::Mat& bgr) {
    cv::Mat g(bgr.size(), CV_8UC1);
    for (int y = 0; y < bgr.rows; ++y)
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b p = bgr.at<cv::Vec3b>(y, x);
            g.at<uchar>(y, x) = cv::saturate_cast<uchar>(static_cast<int>(std::lround(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2])));  // YANLIS: B'ye R agirligi
        }
    return g;
}

struct DiffStats {
    long long sumAbs = 0;
    long long differing = 0;
    int maxDiff = 0;
};

static DiffStats compareImages(const cv::Mat& a, const cv::Mat& b) {
    DiffStats s;
    for (int y = 0; y < a.rows; ++y)
        for (int x = 0; x < a.cols; ++x) {
            const int d = std::abs(static_cast<int>(a.at<uchar>(y, x)) - static_cast<int>(b.at<uchar>(y, x)));
            s.sumAbs += d;
            s.differing += d != 0 ? 1 : 0;
            s.maxDiff = std::max(s.maxDiff, d);
        }
    return s;
}

int main(int argc, char** argv) {
    const fsys::path outDir = (argc >= 2) ? fsys::path(argv[1]) : fsys::path("cikti_histogram_gri");
    const std::string path = (argc >= 3) ? argv[2] : "images/Cat.jpg";
    fsys::create_directories(outDir);

    // ---- Gorev 1: resmi oku, kanallari ayir ----
    const cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "Hata: resim okunamadi: " << path << "\n";
        return 1;
    }
    std::vector<cv::Mat> ch;
    cv::split(bgr, ch);  // ch[0] = B, ch[1] = G, ch[2] = R
    const std::string names[3] = {"Mavi (B)", "Yesil (G)", "Kirmizi (R)"};
    const cv::Scalar colors[3] = {cv::Scalar(220, 80, 0), cv::Scalar(0, 150, 0), cv::Scalar(0, 0, 220)};
    const long long totalPixels = static_cast<long long>(bgr.rows) * bgr.cols;

    std::cout << "Resim: " << path << " (" << bgr.cols << "x" << bgr.rows << ", " << bgr.channels() << " kanal, " << totalPixels << " piksel)\n";
    std::cout << "OpenCV bu resmi B, G, R sirasiyla okur (cikti isimleri: kanal 0 = Mavi, 1 = Yesil, 2 = Kirmizi)\n\n";

    // ---- Gorev 1 ve 2: OpenCV ve kendi histogramlari ----
    std::cout << "GOREV 1 ve 2 - KANAL HISTOGRAMLARI (256 kutu)\n";
    std::cout << "  " << std::left << std::setw(14) << "Kanal" << std::right << std::setw(14) << "toplam piksel" << std::setw(10) << "ortalama" << std::setw(12) << "en yuksek k" << std::setw(16) << "en yuksek sayi"
              << std::setw(22) << "kendi - OpenCV (|fark|)" << "\n";
    std::vector<std::vector<long long>> cvHist(3), myHist(3);
    for (int c = 0; c < 3; ++c) {
        cvHist[c] = openCvCalcHist(ch[c]);
        myHist[c] = myCalcHist(ch[c]);
        long long sum = 0, total = 0, absDiff = 0;
        int peak = 0;
        for (int v = 0; v < 256; ++v) {
            sum += static_cast<long long>(v) * myHist[c][v];
            total += myHist[c][v];
            absDiff += std::llabs(myHist[c][v] - cvHist[c][v]);
            if (myHist[c][v] > myHist[c][peak]) peak = v;
        }
        std::cout << "  " << std::left << std::setw(14) << names[c] << std::right << std::setw(14) << total << std::setw(10) << std::fixed << std::setprecision(2) << static_cast<double>(sum) / total << std::setw(12)
                  << peak << std::setw(16) << myHist[c][peak] << std::setw(22) << absDiff << "\n";
    }
    // farkli kutu sayisi ile de dogrula
    std::cout << "  Farkli kutu sayilariyla dogrulama (3 kanalin toplam mutlak farki; 0 = ayni):";
    for (int bins : {256, 128, 64, 16, 10}) {
        long long d = 0;
        for (int c = 0; c < 3; ++c) {
            const std::vector<long long> a = openCvCalcHist(ch[c], bins), b = myCalcHist(ch[c], bins);
            for (int i = 0; i < bins; ++i) d += std::llabs(a[i] - b[i]);
        }
        std::cout << " " << bins << " kutu: " << d << ";";
    }
    std::cout << "\n\n";

    cv::Mat plotCv, plotMy;
    {
        std::vector<cv::Mat> a, b;
        for (int c = 0; c < 3; ++c) {
            a.push_back(plotHist(cvHist[c], colors[c], "OpenCV calcHist: " + names[c]));
            b.push_back(plotHist(myHist[c], colors[c], "Kendi calcHist: " + names[c]));
        }
        cv::vconcat(a, plotCv);
        cv::vconcat(b, plotMy);
    }
    cv::imwrite((outDir / "1_histogramlar_opencv.png").string(), plotCv);
    cv::imwrite((outDir / "2_histogramlar_kendi.png").string(), plotMy);

    // ---- Gorev 3: gri donusum ----
    cv::Mat grayCv;
    cv::cvtColor(bgr, grayCv, cv::COLOR_BGR2GRAY);

    struct Item {
        std::string name;
        cv::Mat img;
    };
    const std::vector<Item> items = {{"A) kendi, kayan nokta (0.299 R + 0.587 G + 0.114 B)", gri_kayan_nokta(bgr)},
                                     {"B) kendi, 15 bit sabit nokta (9798, 19235, 3735; OpenCV 5.x)", gri_sabit_nokta(bgr)},
                                     {"C) basit ortalama (R + G + B) / 3", gri_ortalama(bgr)},
                                     {"D) yanlis kanal sirasi (B'ye R agirligi)", gri_yanlis_sira(bgr)},
                                     {"E) kendi, 14 bit sabit nokta (4899, 9617, 1868; eski OpenCV)", gri_sabit_nokta14(bgr)}};

    std::cout << "GOREV 3 - GRI DONUSUM: kendi fonksiyonlar ile OpenCV cvtColor(BGR2GRAY) arasindaki fark\n";
    std::cout << "  Hata degeri = eslesen piksellerin mutlak farklarinin toplami (sum |kendi - OpenCV|)\n";
    std::cout << "  " << std::left << std::setw(56) << "Fonksiyon" << std::right << std::setw(16) << "mutlak fark top." << std::setw(14) << "farkli piksel" << std::setw(10) << "oran %" << std::setw(10) << "en buyuk" << "\n";
    for (const Item& it : items) {
        const DiffStats s = compareImages(it.img, grayCv);
        std::cout << "  " << std::left << std::setw(56) << it.name << std::right << std::setw(16) << s.sumAbs << std::setw(14) << s.differing << std::setw(10) << std::setprecision(3)
                  << 100.0 * s.differing / totalPixels << std::setw(10) << s.maxDiff << "\n";
    }
    std::cout << "  (toplam " << totalPixels << " piksel)\n\n";

    cv::imwrite((outDir / "3_gri_opencv.png").string(), grayCv);
    cv::imwrite((outDir / "3_gri_kendi_kayan_nokta.png").string(), items[0].img);
    cv::imwrite((outDir / "3_gri_kendi_sabit_nokta.png").string(), items[1].img);
    {
        cv::Mat d;
        cv::absdiff(items[0].img, grayCv, d);
        cv::Mat d50;
        d.convertTo(d50, CV_8U, 50.0);  // farki gorunur yapmak icin 50 kat
        cv::imwrite((outDir / "3_fark_kayan_nokta_x50.png").string(), d50);
        cv::absdiff(items[3].img, grayCv, d);
        cv::imwrite((outDir / "3_fark_yanlis_sira.png").string(), d);
    }

    std::cout << "Kaydedilenler (" << outDir.string() << "):\n"
              << "  1_histogramlar_opencv.png, 2_histogramlar_kendi.png : B, G, R kanal histogramlari (ust: mavi, orta: yesil, alt: kirmizi)\n"
              << "  3_gri_opencv.png, 3_gri_kendi_kayan_nokta.png, 3_gri_kendi_sabit_nokta.png : gri resimler\n"
              << "  3_fark_kayan_nokta_x50.png (50 kat buyutulmus), 3_fark_yanlis_sira.png : OpenCV gri resmiyle farklar\n";
    return 0;
}

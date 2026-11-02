// Ödev 5 - Görev 5: X-ray (bagaj) görüntülerini gri tonluya çevirip güvenlik operatörleri için renklendirme.
//
// Her görüntü için:
//   1) Kendi gri dönüşüm fonksiyonumuzla (15 bit sabit nokta; OpenCV 5.x'in hesabıyla aynı) gri resim üretilir.
//   2) Gri resim, ÜÇ palet ve bir "en iyi olası" üst sınırla renklendirilir:
//        - Kendi paletimiz (parlaklık korumalı, turuncu - yeşil - mavi): gri düzey g için renk, YCbCr uzayında Y = g
//          alınıp, renk (Cb, Cr) değerleri tarayıcıların malzeme kodlamasına (soluk sarı arka plan, turuncu organik, yeşil karışık,
//          mavi inorganik/metal, koyu mavi kalın metal) göre ayarlanarak oluşturulur. Böylece renkli resmin parlaklığı (gri hali)
//          girdi gri resimle aynıdır: renk ek bilgi olarak eklenir, yoğunluk bilgisi bozulmaz.
//        - OpenCV JET (klasik gökkuşağı), TURBO ve VIRIDIS hazır paletleri (karşılaştırma için).
//        - "Üst sınır" (oracle): orijinal renkli resimden, her gri düzey için o düzeydeki piksellerin ortalama rengi (Lab'de) alınır;
//          bu, gri düzeye dayanan HERHANGİ bir renklendirmenin ulaşabileceği en iyi sonuca yakın bir sınırdır.
//   3) Renklendirmeler, orijinal renkli (scanner'ın malzeme kodlu) resimle karşılaştırılır:
//        - ortalama renk farkı (ΔE76, Lab) ve ΔE > 25 olan piksellerin oranı,
//        - malzeme sınıfı uyumu: pikseller renk tonuna göre 4 sınıfa ayrılır (arka plan/soluk, turuncu, yeşil, mavi) ve
//          orijinaldeki sınıfla aynı olan piksellerin oranı,
//        - parlaklık korunumu: renklendirilmiş resmin gri hali ile girdi gri resim arasındaki ortalama mutlak fark.
//   4) Orijinaldeki her sınıfın (turuncu, yeşil, mavi, arka plan) gri düzey ortalaması ve standart sapması yazdırılır:
//      sınıflar gri düzeyde ne kadar iç içe geçiyor, görülür.
//
// Kullanım: ./xray_renk [cikti_dizini] [girdi_klasoru=images/ThreadsRGB] [en_fazla_goruntu=5]
// Derleme:  g++ -O2 odev5_5_xray_renklendirme.cpp -o xray_renk.exe -std=c++17 $(pkg-config --cflags --libs opencv5)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fsys = std::filesystem;

// ---------------------------------------------------------------------------------------------
// Kendi gri dönüşümümüz (15 bit sabit nokta)
// ---------------------------------------------------------------------------------------------
static cv::Mat grayFixed15(const cv::Mat& bgr) {
    cv::Mat g(bgr.size(), CV_8UC1);
    for (int y = 0; y < bgr.rows; ++y)
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b p = bgr.at<cv::Vec3b>(y, x);
            g.at<uchar>(y, x) = static_cast<uchar>((p[2] * 9798 + p[1] * 19235 + p[0] * 3735 + (1 << 14)) >> 15);
        }
    return g;
}

// ---------------------------------------------------------------------------------------------
// Kendi paletimiz: 256 girişli arama tablosu (indeks = gri düzey, değer = BGR renk)
// ---------------------------------------------------------------------------------------------
static std::vector<cv::Vec3b> makeOwnPalette() {
    struct Anchor {
        double r, g, b;
    };
    // artan parlaklık sırasıyla: koyu mavi, mavi, yeşil, turuncu, soluk sarı
    const Anchor anchors[] = {{15, 45, 160}, {30, 125, 235}, {130, 165, 30}, {240, 140, 30}, {255, 247, 190}};
    struct Stop {
        double y, cb, cr;
    };
    std::vector<Stop> st;
    st.push_back({0.0, 128.0, 128.0});  // siyah: renksiz
    for (const Anchor& a : anchors) {
        const double y = 0.299 * a.r + 0.587 * a.g + 0.114 * a.b;
        const double cb = 128.0 - 0.168736 * a.r - 0.331264 * a.g + 0.5 * a.b;
        const double cr = 128.0 + 0.5 * a.r - 0.418688 * a.g - 0.081312 * a.b;
        st.push_back({y, cb, cr});
    }
    st.push_back({255.0, 128.0, 128.0});  // beyaz: renksiz
    std::vector<cv::Vec3b> lut(256);
    for (int g = 0; g < 256; ++g) {
        size_t i = 0;
        while (i + 2 < st.size() && g > st[i + 1].y) ++i;
        const double t = (g - st[i].y) / (st[i + 1].y - st[i].y);
        const double cb = st[i].cb + t * (st[i + 1].cb - st[i].cb), cr = st[i].cr + t * (st[i + 1].cr - st[i].cr);
        const double R = g + 1.402 * (cr - 128.0), G = g - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0), B = g + 1.772 * (cb - 128.0);
        lut[g] = cv::Vec3b(cv::saturate_cast<uchar>(B), cv::saturate_cast<uchar>(G), cv::saturate_cast<uchar>(R));
    }
    return lut;
}

static cv::Mat applyLut(const cv::Mat& gray, const std::vector<cv::Vec3b>& lut) {
    cv::Mat out(gray.size(), CV_8UC3);
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x) out.at<cv::Vec3b>(y, x) = lut[gray.at<uchar>(y, x)];
    return out;
}

// ---------------------------------------------------------------------------------------------
// Olcumler
// ---------------------------------------------------------------------------------------------
static cv::Mat toLab32(const cv::Mat& bgr8) {
    cv::Mat f, lab;
    bgr8.convertTo(f, CV_32F, 1.0 / 255.0);
    cv::cvtColor(f, lab, cv::COLOR_BGR2Lab);
    return lab;
}

static void deltaE(const cv::Mat& labA, const cv::Mat& labB, double& mean, double& fracOver25) {
    double sum = 0.0;
    long long over = 0;
    for (int y = 0; y < labA.rows; ++y)
        for (int x = 0; x < labA.cols; ++x) {
            const cv::Vec3f a = labA.at<cv::Vec3f>(y, x), b = labB.at<cv::Vec3f>(y, x);
            const double d = std::sqrt(static_cast<double>((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])));
            sum += d;
            if (d > 25.0) ++over;
        }
    const double n = static_cast<double>(labA.rows) * labA.cols;
    mean = sum / n;
    fracOver25 = 100.0 * over / n;
}

// 0: arka plan / soluk, 1: turuncu (organik), 2: yesil (karisik), 3: mavi (inorganik / metal)
static cv::Mat classMap(const cv::Mat& bgr) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat cls(bgr.size(), CV_8UC1);
    for (int y = 0; y < bgr.rows; ++y)
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b p = hsv.at<cv::Vec3b>(y, x);
            const int H = p[0] * 2, S = p[1];
            uchar c;
            if (S < 60) c = 0;
            else if (H < 50 || H >= 330) c = 1;
            else if (H < 150) c = 2;
            else c = 3;
            cls.at<uchar>(y, x) = c;
        }
    return cls;
}

static double classAgreement(const cv::Mat& a, const cv::Mat& b) {
    long long same = 0;
    for (int y = 0; y < a.rows; ++y)
        for (int x = 0; x < a.cols; ++x) same += a.at<uchar>(y, x) == b.at<uchar>(y, x) ? 1 : 0;
    return 100.0 * same / (static_cast<double>(a.rows) * a.cols);
}

static double meanAbsDiff(const cv::Mat& a, const cv::Mat& b) {
    double s = 0.0;
    for (int y = 0; y < a.rows; ++y)
        for (int x = 0; x < a.cols; ++x) s += std::abs(static_cast<int>(a.at<uchar>(y, x)) - static_cast<int>(b.at<uchar>(y, x)));
    return s / (static_cast<double>(a.rows) * a.cols);
}

// Ust sinir: her gri duzey icin orijinalin ortalama Lab rengi
static cv::Mat oracleColorize(const cv::Mat& orig, const cv::Mat& gray) {
    const cv::Mat lab = toLab32(orig);
    std::array<std::array<double, 3>, 256> sum{};
    std::array<long long, 256> cnt{};
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x) {
            const int g = gray.at<uchar>(y, x);
            const cv::Vec3f v = lab.at<cv::Vec3f>(y, x);
            for (int c = 0; c < 3; ++c) sum[g][c] += v[c];
            ++cnt[g];
        }
    std::array<cv::Vec3f, 256> lut;
    std::array<bool, 256> has{};
    for (int g = 0; g < 256; ++g)
        if (cnt[g] > 0) {
            has[g] = true;
            lut[g] = cv::Vec3f(static_cast<float>(sum[g][0] / cnt[g]), static_cast<float>(sum[g][1] / cnt[g]), static_cast<float>(sum[g][2] / cnt[g]));
        }
    for (int g = 0; g < 256; ++g)  // bos duzeyleri en yakin dolu duzeyden doldur
        if (!has[g]) {
            for (int d = 1; d < 256; ++d) {
                if (g - d >= 0 && has[g - d]) { lut[g] = lut[g - d]; break; }
                if (g + d < 256 && has[g + d]) { lut[g] = lut[g + d]; break; }
            }
        }
    cv::Mat labOut(gray.size(), CV_32FC3);
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x) labOut.at<cv::Vec3f>(y, x) = lut[gray.at<uchar>(y, x)];
    cv::Mat bgrF, out;
    cv::cvtColor(labOut, bgrF, cv::COLOR_Lab2BGR);
    bgrF.convertTo(out, CV_8U, 255.0);
    return out;
}

// Karisiklik matrisi: satir = orijinaldeki sinif, sutun = kendi paletimizle renklendirilen resimdeki sinif (satir yuzdesi)
static void printConfusion(const long long conf[4][4], const std::string& title) {
    const char* names[4] = {"arka plan", "turuncu", "yesil", "mavi"};
    std::cout << "  " << title << "\n    " << std::left << std::setw(18) << "orijinal / palet";
    for (int j = 0; j < 4; ++j) std::cout << std::right << std::setw(11) << names[j];
    std::cout << std::setw(14) << "piksel sayisi" << "\n";
    for (int i = 0; i < 4; ++i) {
        long long total = 0;
        for (int j = 0; j < 4; ++j) total += conf[i][j];
        std::cout << "    " << std::left << std::setw(18) << names[i];
        for (int j = 0; j < 4; ++j) std::cout << std::right << std::fixed << std::setprecision(1) << std::setw(11) << (total ? 100.0 * conf[i][j] / total : 0.0);
        std::cout << std::setw(14) << total << "\n";
    }
}

static cv::Mat panel(const cv::Mat& img, const std::string& label, int height) {
    cv::Mat bgr = img.channels() == 1 ? cv::Mat() : img, s;
    if (img.channels() == 1) cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
    const double sc = static_cast<double>(height) / bgr.rows;
    cv::resize(bgr, s, cv::Size(), sc, sc, cv::INTER_AREA);
    cv::putText(s, label, cv::Point(5, 16), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(s, label, cv::Point(5, 16), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return s;
}

int main(int argc, char** argv) {
    const fsys::path outDir = (argc >= 2) ? fsys::path(argv[1]) : fsys::path("cikti_xray");
    const fsys::path inDir = (argc >= 3) ? fsys::path(argv[2]) : fsys::path("images/ThreadsRGB");
    const size_t maxImages = (argc >= 4) ? static_cast<size_t>(std::stoi(argv[3])) : 5;
    fsys::create_directories(outDir);

    std::vector<fsys::path> files;
    if (fsys::exists(inDir))
        for (const auto& e : fsys::directory_iterator(inDir)) {
            std::string ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff") files.push_back(e.path());
        }
    std::sort(files.begin(), files.end());
    if (files.size() > maxImages) files.resize(maxImages);
    if (files.empty()) {
        std::cerr << "Hata: " << inDir.string() << " klasorunde goruntu bulunamadi\n";
        return 1;
    }

    const std::vector<cv::Vec3b> ownLut = makeOwnPalette();
    struct PalInfo {
        std::string name;
        int cvMap;  // -1: kendi paletimiz, -2: ust sinir
        bool invert = false;  // true: gri duzey ters cevrilerek (255 - g) paletlenir
    };
    const std::vector<PalInfo> pals = {{"Kendi paletimiz (parlaklik korumali)", -1},
                                       {"OpenCV JET", cv::COLORMAP_JET},
                                       {"OpenCV TURBO", cv::COLORMAP_TURBO},
                                       {"OpenCV VIRIDIS", cv::COLORMAP_VIRIDIS},
                                       {"OpenCV JET (ters gri)", cv::COLORMAP_JET, true},
                                       {"OpenCV TURBO (ters gri)", cv::COLORMAP_TURBO, true},
                                       {"OpenCV VIRIDIS (ters gri)", cv::COLORMAP_VIRIDIS, true},
                                       {"Ust sinir (gri duzeye gore en iyi renk)", -2}};
    std::vector<double> sumDe(pals.size(), 0.0), sumAgree(pals.size(), 0.0), sumLum(pals.size(), 0.0);

    // palet seritleri
    {
        std::vector<cv::Mat> strips;
        for (const PalInfo& p : pals) {
            if (p.cvMap == -2) continue;
            cv::Mat grad(1, 256, CV_8UC1);
            for (int i = 0; i < 256; ++i) grad.at<uchar>(0, i) = static_cast<uchar>(i);
            const cv::Mat gsrc = p.invert ? cv::Mat(255 - grad) : grad;
            cv::Mat col = p.cvMap == -1 ? applyLut(grad, ownLut) : cv::Mat();
            if (p.cvMap != -1) cv::applyColorMap(gsrc, col, p.cvMap);
            cv::Mat big;
            cv::resize(col, big, cv::Size(768, 50), 0, 0, cv::INTER_NEAREST);
            cv::putText(big, p.name + "  (sol: koyu / yogun, sag: acik / seffaf)", cv::Point(5, 16), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            cv::putText(big, p.name + "  (sol: koyu / yogun, sag: acik / seffaf)", cv::Point(5, 16), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            strips.push_back(big);
        }
        cv::Mat all;
        cv::vconcat(strips, all);
        cv::imwrite((outDir / "5_palet_seritleri.png").string(), all);
    }

    std::cout << "Girdi klasoru: " << inDir.string() << " (" << files.size() << " goruntu)\n";
    int idx = 0;
    long long confAll[4][4] = {};
    for (const fsys::path& f : files) {
        ++idx;
        const cv::Mat orig = cv::imread(f.string(), cv::IMREAD_COLOR);
        if (orig.empty()) {
            std::cout << "\n[" << idx << "] okunamadi: " << f.string() << "\n";
            continue;
        }
        const cv::Mat gray = grayFixed15(orig);
        const cv::Mat clsOrig = classMap(orig);
        const cv::Mat labOrig = toLab32(orig);
        const long long N = static_cast<long long>(orig.rows) * orig.cols;

        std::cout << "\n=== [" << idx << "] " << f.filename().string() << " (" << orig.cols << "x" << orig.rows << ") ===\n";

        // sinif - gri duzey istatistigi
        {
            const char* names[4] = {"arka plan / soluk", "turuncu (organik)", "yesil (karisik)", "mavi (inorganik / metal)"};
            double s[4] = {0, 0, 0, 0}, s2[4] = {0, 0, 0, 0};
            long long n[4] = {0, 0, 0, 0};
            for (int y = 0; y < orig.rows; ++y)
                for (int x = 0; x < orig.cols; ++x) {
                    const int c = clsOrig.at<uchar>(y, x);
                    const double g = gray.at<uchar>(y, x);
                    s[c] += g;
                    s2[c] += g * g;
                    ++n[c];
                }
            std::cout << "  Orijinaldeki renk sinifi ve gri duzeyi (sinif, orijinalin renk tonuna gore belirlendi):\n";
            std::cout << "    " << std::left << std::setw(28) << "sinif" << std::right << std::setw(12) << "piksel %" << std::setw(12) << "gri ort." << std::setw(12) << "gri std" << "\n";
            for (int c = 0; c < 4; ++c) {
                const double mean = n[c] ? s[c] / n[c] : 0.0, sd = n[c] ? std::sqrt(std::max(0.0, s2[c] / n[c] - mean * mean)) : 0.0;
                std::cout << "    " << std::left << std::setw(28) << names[c] << std::right << std::fixed << std::setprecision(2) << std::setw(12) << 100.0 * n[c] / N << std::setprecision(1) << std::setw(12) << mean
                          << std::setw(12) << sd << "\n";
            }
        }

        // renklendirmeler
        std::vector<cv::Mat> colored;
        for (const PalInfo& p : pals) {
            if (p.cvMap == -1) colored.push_back(applyLut(gray, ownLut));
            else if (p.cvMap == -2) colored.push_back(oracleColorize(orig, gray));
            else {
                cv::Mat c;
                const cv::Mat src = p.invert ? cv::Mat(255 - gray) : gray;
                cv::applyColorMap(src, c, p.cvMap);
                colored.push_back(c);
            }
        }
        std::cout << "  Renklendirmelerin orijinal renkli resimle karsilastirmasi:\n";
        std::cout << "    " << std::left << std::setw(42) << "Palet" << std::right << std::setw(12) << "ort. dE76" << std::setw(14) << "dE>25 piksel%" << std::setw(16) << "sinif uyumu %" << std::setw(20) << "parlaklik sapmasi" << "\n";
        for (size_t k = 0; k < pals.size(); ++k) {
            double de, over;
            deltaE(toLab32(colored[k]), labOrig, de, over);
            const double agree = classAgreement(classMap(colored[k]), clsOrig);
            const double lum = meanAbsDiff(grayFixed15(colored[k]), gray);
            std::cout << "    " << std::left << std::setw(42) << pals[k].name << std::right << std::fixed << std::setprecision(2) << std::setw(12) << de << std::setw(14) << over << std::setw(16) << agree << std::setw(20)
                      << lum << "\n";
            sumDe[k] += de;
            sumAgree[k] += agree;
            sumLum[k] += lum;
        }

        // karisiklik matrisi (kendi paletimiz)
        {
            const cv::Mat clsOwn = classMap(colored[0]);
            long long conf[4][4] = {};
            for (int y = 0; y < orig.rows; ++y)
                for (int x = 0; x < orig.cols; ++x) ++conf[clsOrig.at<uchar>(y, x)][clsOwn.at<uchar>(y, x)];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) confAll[i][j] += conf[i][j];
            printConfusion(conf, "Kendi paletimiz icin sinif karisiklik matrisi (satir yuzdesi):");
        }

        // gorseller
        const std::string tag = "5_goruntu" + std::to_string(idx);
        cv::imwrite((outDir / (tag + "_gri.png")).string(), gray);
        cv::imwrite((outDir / (tag + "_renkli_kendi_palet.png")).string(), colored[0]);
        std::vector<cv::Mat> panels = {panel(orig, "orijinal (scanner renk kodlu)", 240), panel(gray, "gri (kendi donusumumuz)", 240)};
        for (size_t k = 0; k < pals.size(); ++k) panels.push_back(panel(colored[k], pals[k].name.substr(0, 28), 240));
        cv::Mat row1, row2, all;
        const size_t half = (panels.size() + 1) / 2;
        cv::hconcat(std::vector<cv::Mat>(panels.begin(), panels.begin() + half), row1);
        std::vector<cv::Mat> rest(panels.begin() + half, panels.end());
        cv::hconcat(rest, row2);
        if (row2.cols < row1.cols) cv::copyMakeBorder(row2, row2, 0, 0, 0, row1.cols - row2.cols, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
        if (row1.cols < row2.cols) cv::copyMakeBorder(row1, row1, 0, 0, 0, row2.cols - row1.cols, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
        cv::vconcat(std::vector<cv::Mat>{row1, row2}, all);
        cv::imwrite((outDir / (tag + "_karsilastirma.png")).string(), all);
    }

    std::cout << "\n=== TUM GORUNTULERIN ORTALAMASI (" << idx << " goruntu) ===\n";
    std::cout << "    " << std::left << std::setw(42) << "Palet" << std::right << std::setw(12) << "ort. dE76" << std::setw(16) << "sinif uyumu %" << std::setw(20) << "parlaklik sapmasi" << "\n";
    for (size_t k = 0; k < pals.size(); ++k)
        std::cout << "    " << std::left << std::setw(42) << pals[k].name << std::right << std::fixed << std::setprecision(2) << std::setw(12) << sumDe[k] / idx << std::setw(16) << sumAgree[k] / idx << std::setw(20)
                  << sumLum[k] / idx << "\n";
    printConfusion(confAll, "\nTUM GORUNTULER: kendi paletimiz icin sinif karisiklik matrisi (satir yuzdesi):");
    std::cout << "\nKaydedilenler (" << outDir.string() << "):\n"
              << "  5_goruntuK_gri.png, 5_goruntuK_renkli_kendi_palet.png : K. goruntunun gri hali ve kendi paletimizle renklendirilmis hali\n"
              << "  5_goruntuK_karsilastirma.png                          : orijinal, gri ve tum renklendirmeler yan yana\n"
              << "  5_palet_seritleri.png                                 : paletlerin gri duzeyden renge eslemesi\n";
    return 0;
}

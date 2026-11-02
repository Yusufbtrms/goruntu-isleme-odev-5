// Ödev 5 - Görev 4: Renk uzayı dönüşüm fonksiyonları (fonksiyon göstericileriyle) ve OpenCV ile sayısal karşılaştırma.
//
// Dönüşümler (PDF'teki formüller): BGR <-> YCrCb, BGR <-> HSV, BGR <-> CIELAB, BGR <-> HSI, BGR -> normalize rgb, BGR -> gri.
// OpenCV'de HSI ve normalize rgb için hazır fonksiyon yoktur: HSI bilinen renklerle ve geri dönüşümle, normalize rgb OpenCV
// matris işlemleriyle kurulan bir referansla doğrulanır.
//
// ÖNEMLİ (neden iki türlü fonksiyon var?): Ders kitabı formüllerini aynen uygulamak (A tipi, "PDF formülü") OpenCV'nin 8-bit
// sonucundan bazı piksellerde 1 seviye farklı çıkabilir, çünkü OpenCV 8-bit resimlerde bu formülleri sabit noktalı tam sayı
// aritmetiğiyle ve (Lab'de) arama tablolarıyla hesaplar; katsayıları da formüldekinin bire bir aynısı değildir (örneğin Cr için
// 0.713, Cb için 0.564). "Beklenen hata sıfırdır" sonucuna ulaşmak için, OpenCV'nin hesabını birebir taklit eden B tipi
// fonksiyonlar da yazıldı. Her iki türün hatası ayrı ayrı yazdırılır.
//
// Kullanım: ./renk_uzaylari [cikti_dizini] [resim_yolu=images/Cat.jpg] [en_uzun_kenar=800 (0 = kucultme yok)]
// Derleme:  g++ -O2 odev5_4_renk_uzaylari.cpp -o renk_uzaylari.exe -std=c++17 $(pkg-config --cflags --libs opencv5)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fsys = std::filesystem;

// Fonksiyon göstericisi türü: bir resmi alıp başka bir renk uzayındaki resmi döndüren fonksiyon
using ConvFn = cv::Mat (*)(const cv::Mat&);

static const double PI = 3.14159265358979323846;

static inline int descale(int x, int n) { return (x + (1 << (n - 1))) >> n; }  // OpenCV'nin CV_DESCALE'i
static inline uchar clampU8(int v) { return static_cast<uchar>(v < 0 ? 0 : (v > 255 ? 255 : v)); }

// ---------------------------------------------------------------------------------------------
// Gri. Bu OpenCV sürümünün (5.x) hesabı 15 bit sabit noktadır (bgr2gray_fixed15, sıfır hata verir);
// aşağıdaki 14 bit sabit nokta eski sürümlerin hesabıdır ve bazı piksellerde 1 seviye farklı çıkar.
// ---------------------------------------------------------------------------------------------
// Bu OpenCV sürümünde cvtColor(BGR2GRAY) hangi aritmetiği kullanıyor? Üç aday:
//   15 bit sabit nokta (9798, 19235, 3735), float32 (0.299f, 0.587f, 0.114f) ve çift duyarlıklı kayan nokta
static cv::Mat bgr2gray_fixed15(const cv::Mat& src) {
    cv::Mat out(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            out.at<uchar>(y, x) = static_cast<uchar>(descale(p[2] * 9798 + p[1] * 19235 + p[0] * 3735, 15));
        }
    return out;
}
static cv::Mat bgr2gray_float32(const cv::Mat& src) {
    cv::Mat out(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            const float g = p[0] * 0.114f + p[1] * 0.587f + p[2] * 0.299f;
            out.at<uchar>(y, x) = cv::saturate_cast<uchar>(g);
        }
    return out;
}
static cv::Mat bgr2gray_double(const cv::Mat& src) {
    cv::Mat out(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            out.at<uchar>(y, x) = cv::saturate_cast<uchar>(0.299 * p[2] + 0.587 * p[1] + 0.114 * p[0]);
        }
    return out;
}

static cv::Mat bgr2gray_exact(const cv::Mat& src) {
    cv::Mat out(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            out.at<uchar>(y, x) = static_cast<uchar>(descale(p[2] * 4899 + p[1] * 9617 + p[0] * 1868, 14));
        }
    return out;
}

// ---------------------------------------------------------------------------------------------
// YCrCb  (OpenCV kanal sırası: Y, Cr, Cb)
// ---------------------------------------------------------------------------------------------
// A) PDF formülü (BT.601, tam aralık)
static cv::Mat bgr2ycrcb_textbook(const cv::Mat& src) {
    cv::Mat out(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            const double B = p[0], G = p[1], R = p[2];
            const double Y = 0.299 * R + 0.587 * G + 0.114 * B;
            const double Cb = 128.0 - 0.168736 * R - 0.331264 * G + 0.5 * B;
            const double Cr = 128.0 + 0.5 * R - 0.418688 * G - 0.081312 * B;
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(cv::saturate_cast<uchar>(Y), cv::saturate_cast<uchar>(Cr), cv::saturate_cast<uchar>(Cb));
        }
    return out;
}

// B) OpenCV birebir (14 bit sabit nokta; Cr katsayısı 0.713 = 11682/16384, Cb katsayısı 0.564 = 9241/16384)
static cv::Mat bgr2ycrcb_exact(const cv::Mat& src) {
    cv::Mat out(src.size(), CV_8UC3);
    const int delta = 128 << 14;
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            const int B = p[0], G = p[1], R = p[2];
            const int Y = descale(R * 4899 + G * 9617 + B * 1868, 14);
            const int Cr = descale((R - Y) * 11682 + delta, 14);
            const int Cb = descale((B - Y) * 9241 + delta, 14);
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(clampU8(Y), clampU8(Cr), clampU8(Cb));
        }
    return out;
}

static cv::Mat ycrcb2bgr_textbook(const cv::Mat& src) {
    cv::Mat out(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            const double Y = p[0], Cr = p[1] - 128.0, Cb = p[2] - 128.0;
            const double R = Y + 1.402 * Cr, G = Y - 0.344136 * Cb - 0.714136 * Cr, B = Y + 1.772 * Cb;
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(cv::saturate_cast<uchar>(B), cv::saturate_cast<uchar>(G), cv::saturate_cast<uchar>(R));
        }
    return out;
}

static cv::Mat ycrcb2bgr_exact(const cv::Mat& src) {
    cv::Mat out(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            const int Y = p[0], Cr = p[1] - 128, Cb = p[2] - 128;
            const int b = Y + descale(Cb * 29049, 14);
            const int g = Y + descale(Cb * (-5636) + Cr * (-11698), 14);
            const int r = Y + descale(Cr * 22987, 14);
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(clampU8(b), clampU8(g), clampU8(r));
        }
    return out;
}

// ---------------------------------------------------------------------------------------------
// HSV  (OpenCV 8-bit: H = derece / 2, 0..179; S ve V 0..255)
// ---------------------------------------------------------------------------------------------
// A) PDF formülü
static cv::Mat bgr2hsv_textbook(const cv::Mat& src) {
    cv::Mat out(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            const double B = p[0], G = p[1], R = p[2];
            const double M = std::max(R, std::max(G, B)), m = std::min(R, std::min(G, B)), d = M - m;
            const double V = M, S = M > 0.0 ? d / M : 0.0;
            double H = 0.0;
            if (d > 0.0) {
                if (M == R) {
                    H = 60.0 * std::fmod((G - B) / d, 6.0);
                    if (H < 0.0) H += 360.0;
                } else if (M == G) {
                    H = 60.0 * ((B - R) / d + 2.0);
                } else {
                    H = 60.0 * ((R - G) / d + 4.0);
                }
            }
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(cv::saturate_cast<uchar>(H / 2.0), cv::saturate_cast<uchar>(S * 255.0), cv::saturate_cast<uchar>(V));
        }
    return out;
}

// B) OpenCV birebir (12 bit sabit nokta, bölme tabloları)
static cv::Mat bgr2hsv_exact(const cv::Mat& src) {
    static int sdiv[256], hdiv[256];
    static bool init = false;
    if (!init) {
        sdiv[0] = hdiv[0] = 0;
        for (int i = 1; i < 256; ++i) {
            sdiv[i] = cvRound((255 << 12) / (1. * i));
            hdiv[i] = cvRound((180 << 12) / (6. * i));
        }
        init = true;
    }
    cv::Mat out(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            const int b = p[0], g = p[1], r = p[2];
            const int v = std::max(b, std::max(g, r)), vmin = std::min(b, std::min(g, r));
            const int diff = v - vmin;
            const int vr = (v == r) ? -1 : 0, vg = (v == g) ? -1 : 0;
            const int s = (diff * sdiv[v] + (1 << 11)) >> 12;
            int h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
            h = (h * hdiv[diff] + (1 << 11)) >> 12;
            h += h < 0 ? 180 : 0;
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(clampU8(h), clampU8(s), clampU8(v));
        }
    return out;
}

static cv::Mat hsv2bgr_textbook(const cv::Mat& src) {
    cv::Mat out(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            const double H = p[0] * 2.0, S = p[1] / 255.0, V = p[2] / 255.0;
            const double C = V * S, hp = H / 60.0;
            const double X = C * (1.0 - std::fabs(std::fmod(hp, 2.0) - 1.0));
            double r = 0, g = 0, b = 0;
            if (hp < 1) { r = C; g = X; }
            else if (hp < 2) { r = X; g = C; }
            else if (hp < 3) { g = C; b = X; }
            else if (hp < 4) { g = X; b = C; }
            else if (hp < 5) { r = X; b = C; }
            else { r = C; b = X; }
            const double m = V - C;
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(cv::saturate_cast<uchar>((b + m) * 255.0), cv::saturate_cast<uchar>((g + m) * 255.0), cv::saturate_cast<uchar>((r + m) * 255.0));
        }
    return out;
}

// B) OpenCV'nin float tabanlı hesabı birebir
static cv::Mat hsv2bgr_exact(const cv::Mat& src) {
    static const int sector_data[][3] = {{1, 3, 0}, {1, 0, 2}, {3, 0, 1}, {0, 2, 1}, {0, 1, 3}, {2, 1, 0}};
    const float hscale = 6.f / 180.f;
    cv::Mat out(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            float h = p[0], s = p[1] * (1.f / 255.f), v = p[2] * (1.f / 255.f);
            float b, g, r;
            if (s == 0) {
                b = g = r = v;
            } else {
                float tab[4];
                h *= hscale;
                if (h < 0) do h += 6; while (h < 0);
                else if (h >= 6) do h -= 6; while (h >= 6);
                int sector = cvFloor(h);
                h -= sector;
                if (static_cast<unsigned>(sector) >= 6u) { sector = 0; h = 0.f; }
                tab[0] = v;
                tab[1] = v * (1.f - s);
                tab[2] = v * (1.f - s * h);
                tab[3] = v * (1.f - s * (1.f - h));
                b = tab[sector_data[sector][0]];
                g = tab[sector_data[sector][1]];
                r = tab[sector_data[sector][2]];
            }
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(cv::saturate_cast<uchar>(b * 255.f), cv::saturate_cast<uchar>(g * 255.f), cv::saturate_cast<uchar>(r * 255.f));
        }
    return out;
}

// Hipotez: OpenCV bu sürümde float -> 8-bit çevirisinde yuvarlamak yerine KESİYOR (aşağı yuvarlıyor).
// hsv2bgr_exact ile aynı hesap; yalnızca son çevirme farklı.
static cv::Mat hsv2bgr_trunc(const cv::Mat& src) {
    static const int sector_data[][3] = {{1, 3, 0}, {1, 0, 2}, {3, 0, 1}, {0, 2, 1}, {0, 1, 3}, {2, 1, 0}};
    const float hscale = 6.f / 180.f;
    cv::Mat out(src.size(), CV_8UC3);
    auto toU8 = [](float v) { return cv::saturate_cast<uchar>(static_cast<int>(v * 255.f)); };  // kesme
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            float h = p[0], s = p[1] * (1.f / 255.f), v = p[2] * (1.f / 255.f);
            float b, g, r;
            if (s == 0) {
                b = g = r = v;
            } else {
                float tab[4];
                h *= hscale;
                if (h < 0) do h += 6; while (h < 0);
                else if (h >= 6) do h -= 6; while (h >= 6);
                int sector = cvFloor(h);
                h -= sector;
                if (static_cast<unsigned>(sector) >= 6u) { sector = 0; h = 0.f; }
                tab[0] = v;
                tab[1] = v * (1.f - s);
                tab[2] = v * (1.f - s * h);
                tab[3] = v * (1.f - s * (1.f - h));
                b = tab[sector_data[sector][0]];
                g = tab[sector_data[sector][1]];
                r = tab[sector_data[sector][2]];
            }
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(toU8(b), toU8(g), toU8(r));
        }
    return out;
}

// ---------------------------------------------------------------------------------------------
// CIELAB  (PDF: sRGB doğrusallaştırma, D65 XYZ matrisi, beyaz nokta Xn = 0.95047, Yn = 1, Zn = 1.08883)
// 8-bit gösterim OpenCV gibi: L * 255 / 100, a + 128, b + 128
// ---------------------------------------------------------------------------------------------
static double srgbToLinear(double c255) {
    const double c = c255 / 255.0;
    return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}
static double linearToSrgb255(double l) {
    const double c = l <= 0.0031308 ? 12.92 * l : 1.055 * std::pow(l, 1.0 / 2.4) - 0.055;
    return c * 255.0;
}
static double labF(double t) {
    const double e = 216.0 / 24389.0;  // (6/29)^3
    return t > e ? std::cbrt(t) : t / (108.0 / 841.0) + 4.0 / 29.0;  // 3 (6/29)^2 = 108/841
}
static double labFinv(double t) {
    const double d = 6.0 / 29.0;
    return t > d ? t * t * t : (108.0 / 841.0) * (t - 4.0 / 29.0);
}

static cv::Mat bgr2lab_textbook(const cv::Mat& src) {
    const double Xn = 0.95047, Yn = 1.0, Zn = 1.08883;
    cv::Mat out(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            const double R = srgbToLinear(p[2]), G = srgbToLinear(p[1]), B = srgbToLinear(p[0]);
            const double X = 0.4124 * R + 0.3576 * G + 0.1805 * B;
            const double Y = 0.2126 * R + 0.7152 * G + 0.0722 * B;
            const double Z = 0.0193 * R + 0.1192 * G + 0.9505 * B;
            const double fx = labF(X / Xn), fy = labF(Y / Yn), fz = labF(Z / Zn);
            const double L = 116.0 * fy - 16.0, a = 500.0 * (fx - fy), b = 200.0 * (fy - fz);
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(cv::saturate_cast<uchar>(L * 255.0 / 100.0), cv::saturate_cast<uchar>(a + 128.0), cv::saturate_cast<uchar>(b + 128.0));
        }
    return out;
}

static cv::Mat lab2bgr_textbook(const cv::Mat& src) {
    const double Xn = 0.95047, Yn = 1.0, Zn = 1.08883;
    const cv::Matx33d M(0.4124, 0.3576, 0.1805, 0.2126, 0.7152, 0.0722, 0.0193, 0.1192, 0.9505);
    const cv::Matx33d Minv = M.inv();
    cv::Mat out(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            const double L = p[0] * 100.0 / 255.0, a = p[1] - 128.0, b = p[2] - 128.0;
            const double fy = (L + 16.0) / 116.0, fx = fy + a / 500.0, fz = fy - b / 200.0;
            const double X = Xn * labFinv(fx), Y = Yn * labFinv(fy), Z = Zn * labFinv(fz);
            const double r = Minv(0, 0) * X + Minv(0, 1) * Y + Minv(0, 2) * Z;
            const double g = Minv(1, 0) * X + Minv(1, 1) * Y + Minv(1, 2) * Z;
            const double bl = Minv(2, 0) * X + Minv(2, 1) * Y + Minv(2, 2) * Z;
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(cv::saturate_cast<uchar>(linearToSrgb255(std::max(0.0, bl))), cv::saturate_cast<uchar>(linearToSrgb255(std::max(0.0, g))),
                                                cv::saturate_cast<uchar>(linearToSrgb255(std::max(0.0, r))));
        }
    return out;
}

// ---------------------------------------------------------------------------------------------
// HSI (Gonzalez & Woods)  ve normalize rgb  (cikti: CV_32FC3)
//   HSI cikti kanallari: H (derece), S, I   |   normalize rgb cikti kanallari: r, g, b
// ---------------------------------------------------------------------------------------------
static cv::Mat bgr2hsi(const cv::Mat& src) {
    cv::Mat out(src.size(), CV_32FC3);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            const double R = p[2] / 255.0, G = p[1] / 255.0, B = p[0] / 255.0;
            const double sum = R + G + B, I = sum / 3.0;
            const double S = sum > 0.0 ? 1.0 - 3.0 * std::min(R, std::min(G, B)) / sum : 0.0;  // siyahta tanimsiz: 0 alinir
            const double num = 0.5 * ((R - G) + (R - B));
            const double den = std::sqrt((R - G) * (R - G) + (R - B) * (G - B));
            double H = 0.0;  // gri piksellerde tanimsiz: 0 alinir
            if (den > 1e-12) {
                const double theta = std::acos(std::max(-1.0, std::min(1.0, num / den))) * 180.0 / PI;
                H = (B <= G) ? theta : 360.0 - theta;
            }
            out.at<cv::Vec3f>(y, x) = cv::Vec3f(static_cast<float>(H), static_cast<float>(S), static_cast<float>(I));
        }
    return out;
}

static cv::Mat hsi2bgr(const cv::Mat& src) {
    cv::Mat out(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3f p = src.at<cv::Vec3f>(y, x);
            double H = p[0], S = p[1], I = p[2], R, G, B;
            const double rad = PI / 180.0;
            if (H < 120.0) {
                B = I * (1.0 - S);
                R = I * (1.0 + S * std::cos(H * rad) / std::cos((60.0 - H) * rad));
                G = 3.0 * I - (R + B);
            } else if (H < 240.0) {
                const double h = H - 120.0;
                R = I * (1.0 - S);
                G = I * (1.0 + S * std::cos(h * rad) / std::cos((60.0 - h) * rad));
                B = 3.0 * I - (R + G);
            } else {
                const double h = H - 240.0;
                G = I * (1.0 - S);
                B = I * (1.0 + S * std::cos(h * rad) / std::cos((60.0 - h) * rad));
                R = 3.0 * I - (G + B);
            }
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(cv::saturate_cast<uchar>(B * 255.0), cv::saturate_cast<uchar>(G * 255.0), cv::saturate_cast<uchar>(R * 255.0));
        }
    return out;
}

static cv::Mat bgr2nrgb(const cv::Mat& src) {
    cv::Mat out(src.size(), CV_32FC3);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            const cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            const float R = p[2], G = p[1], B = p[0], S = R + G + B;
            out.at<cv::Vec3f>(y, x) = S > 0.f ? cv::Vec3f(R / S, G / S, B / S) : cv::Vec3f(0.f, 0.f, 0.f);  // siyahta tanimsiz: 0
        }
    return out;
}

// OpenCV matris islemleriyle kurulan referans (normalize rgb)
static cv::Mat nrgb_reference(const cv::Mat& src) {
    cv::Mat f;
    src.convertTo(f, CV_32F);
    std::vector<cv::Mat> ch;
    cv::split(f, ch);  // B, G, R
    cv::Mat S = ch[0] + ch[1] + ch[2], r, g, b;
    cv::divide(ch[2], S, r);  // bolen 0 ise sonuc 0
    cv::divide(ch[1], S, g);
    cv::divide(ch[0], S, b);
    const cv::Mat zeroMask = (S == 0);  // float'ta 0/0 NaN verir; siyah piksellerde (tanimsiz) 0 alinir
    r.setTo(0, zeroMask);
    g.setTo(0, zeroMask);
    b.setTo(0, zeroMask);
    cv::Mat out;
    cv::merge(std::vector<cv::Mat>{r, g, b}, out);
    return out;
}

static cv::Mat identityConv(const cv::Mat& m) { return m.clone(); }

// ---------------------------------------------------------------------------------------------
// Karsilastirma
// ---------------------------------------------------------------------------------------------
struct CmpStats {
    double sumAbs = 0.0, maxDiff = 0.0;
    long long differing = 0;
    std::vector<double> perChannel;
};

static CmpStats compareMats(const cv::Mat& a, const cv::Mat& b, double tol = 0.0) {
    CV_Assert(a.size() == b.size() && a.channels() == b.channels());
    cv::Mat a64, b64;
    a.convertTo(a64, CV_64F);
    b.convertTo(b64, CV_64F);
    const int cn = a.channels();
    CmpStats s;
    s.perChannel.assign(cn, 0.0);
    for (int y = 0; y < a.rows; ++y) {
        const double* pa = a64.ptr<double>(y);
        const double* pb = b64.ptr<double>(y);
        for (int x = 0; x < a.cols; ++x) {
            bool diff = false;
            for (int c = 0; c < cn; ++c) {
                const double d = std::fabs(pa[x * cn + c] - pb[x * cn + c]);
                s.perChannel[c] += d;
                s.sumAbs += d;
                s.maxDiff = std::max(s.maxDiff, d);
                if (!(d <= tol)) diff = true;  // NaN da fark sayilir
            }
            if (diff) ++s.differing;
        }
    }
    return s;
}

static void printCmp(const std::string& name, const CmpStats& s, long long pixels, bool isFloat = false) {
    std::cout << "  " << std::left << std::setw(46) << name << std::right << std::fixed << std::setprecision(isFloat ? 6 : 0) << std::setw(16) << s.sumAbs << std::setw(12) << s.differing
              << std::setprecision(3) << std::setw(9) << 100.0 * s.differing / pixels << std::setprecision(isFloat ? 6 : 0) << std::setw(10) << s.maxDiff << "   kanal:";
    for (double v : s.perChannel) std::cout << " " << v;
    std::cout << "\n";
}

static void printHeader(const std::string& title) {
    std::cout << title << "\n  " << std::left << std::setw(46) << "Donusum" << std::right << std::setw(16) << "mutlak fark top." << std::setw(12) << "farkli piksel" << std::setw(9) << "oran %" << std::setw(10) << "en buyuk"
              << "\n";
}

struct Fwd {
    const char* name;
    ConvFn fn;
    int cvCode;
};
struct Inv {
    const char* name;
    ConvFn fn;
    int cvForward;  // OpenCV'nin ileri donusumu (girdiyi uretmek icin)
    int cvInverse;  // OpenCV'nin ters donusumu (referans)
};
struct RoundTrip {
    const char* name;
    ConvFn fwd;
    ConvFn inv;
    int cvForward;  // -1: OpenCV'de yok
    int cvInverse;
};

int main(int argc, char** argv) {
    const fsys::path outDir = (argc >= 2) ? fsys::path(argv[1]) : fsys::path("cikti_renk_uzaylari");
    const std::string path = (argc >= 3) ? argv[2] : "images/Cat.jpg";
    const int maxSide = (argc >= 4) ? std::stoi(argv[3]) : 800;
    fsys::create_directories(outDir);

    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "Hata: resim okunamadi: " << path << "\n";
        return 1;
    }
    if (maxSide > 0 && std::max(bgr.cols, bgr.rows) > maxSide) {
        const double s = static_cast<double>(maxSide) / std::max(bgr.cols, bgr.rows);
        cv::resize(bgr, bgr, cv::Size(), s, s, cv::INTER_AREA);
    }
    const long long pixels = static_cast<long long>(bgr.rows) * bgr.cols;
    std::cout << "Resim: " << path << " (" << bgr.cols << "x" << bgr.rows << ", " << pixels << " piksel, OpenCV B-G-R sirasi)\n\n";

    // =========================================================================================
    // 1) Ileri donusumler: BGR -> X  (fonksiyon gostericileri tablosu)
    // =========================================================================================
    const Fwd forward[] = {{"Gri: 14 bit sabit nokta", bgr2gray_exact, cv::COLOR_BGR2GRAY},
                           {"Gri: 15 bit sabit nokta", bgr2gray_fixed15, cv::COLOR_BGR2GRAY},
                           {"Gri: float32 (0.299f, 0.587f, 0.114f)", bgr2gray_float32, cv::COLOR_BGR2GRAY},
                           {"Gri: double (0.299, 0.587, 0.114)", bgr2gray_double, cv::COLOR_BGR2GRAY},
                           {"YCrCb: PDF formulu", bgr2ycrcb_textbook, cv::COLOR_BGR2YCrCb},
                           {"YCrCb: OpenCV birebir", bgr2ycrcb_exact, cv::COLOR_BGR2YCrCb},
                           {"HSV: PDF formulu", bgr2hsv_textbook, cv::COLOR_BGR2HSV},
                           {"HSV: OpenCV birebir", bgr2hsv_exact, cv::COLOR_BGR2HSV},
                           {"Lab: PDF formulu (OpenCV'de tablo/sabit nokta)", bgr2lab_textbook, cv::COLOR_BGR2Lab}};
    printHeader("1) ILERI DONUSUMLER (BGR -> X): kendi fonksiyon (gostericiyle cagrildi) ile OpenCV cvtColor arasindaki fark");
    for (const Fwd& e : forward) {
        ConvFn fn = e.fn;  // fonksiyon gostericisi uzerinden cagri
        const cv::Mat mine = fn(bgr);
        cv::Mat ref;
        cv::cvtColor(bgr, ref, e.cvCode);
        printCmp(e.name, compareMats(mine, ref), pixels);
    }
    {
        const cv::Mat mine = bgr2nrgb(bgr), ref = nrgb_reference(bgr);
        printCmp("Normalize rgb (float; OpenCV matris islemleri referansi)", compareMats(mine, ref, 1e-7), pixels, true);
    }
    std::cout << "  (OpenCV'de HSI icin hazir fonksiyon yok; asagidaki bilinen renk ve geri donusum testleriyle dogrulanir)\n\n";

    // =========================================================================================
    // 2) Ters donusumler: X -> BGR  (girdi: OpenCV'nin ileri donusumu; referans: OpenCV'nin ters donusumu)
    // =========================================================================================
    const Inv inverse[] = {{"YCrCb -> BGR: PDF formulu", ycrcb2bgr_textbook, cv::COLOR_BGR2YCrCb, cv::COLOR_YCrCb2BGR},
                           {"YCrCb -> BGR: OpenCV birebir", ycrcb2bgr_exact, cv::COLOR_BGR2YCrCb, cv::COLOR_YCrCb2BGR},
                           {"HSV -> BGR: ders kitabi formulu", hsv2bgr_textbook, cv::COLOR_BGR2HSV, cv::COLOR_HSV2BGR},
                           {"HSV -> BGR: float, yuvarlamali (OpenCV tahmini 1)", hsv2bgr_exact, cv::COLOR_BGR2HSV, cv::COLOR_HSV2BGR},
                           {"HSV -> BGR: float, kesmeli (OpenCV tahmini 2)", hsv2bgr_trunc, cv::COLOR_BGR2HSV, cv::COLOR_HSV2BGR},
                           {"Lab -> BGR: PDF formulu", lab2bgr_textbook, cv::COLOR_BGR2Lab, cv::COLOR_Lab2BGR}};
    printHeader("2) TERS DONUSUMLER (X -> BGR): ayni girdi uzerinde kendi fonksiyon ile OpenCV'nin ters donusumu");
    for (const Inv& e : inverse) {
        cv::Mat input;
        cv::cvtColor(bgr, input, e.cvForward);
        ConvFn fn = e.fn;
        const cv::Mat mine = fn(input);
        cv::Mat ref;
        cv::cvtColor(input, ref, e.cvInverse);
        printCmp(e.name, compareMats(mine, ref), pixels);
    }
    std::cout << "\n";

    // =========================================================================================
    // 3) Gidis-donus: BGR -> X -> BGR, orijinale gore hata (8-bit donusumler kayiplidir)
    // =========================================================================================
    const RoundTrip trips[] = {{"YCrCb (OpenCV birebir fonksiyonlarimiz)", bgr2ycrcb_exact, ycrcb2bgr_exact, cv::COLOR_BGR2YCrCb, cv::COLOR_YCrCb2BGR},
                               {"HSV (OpenCV birebir ileri + kesmeli ters)", bgr2hsv_exact, hsv2bgr_trunc, cv::COLOR_BGR2HSV, cv::COLOR_HSV2BGR},
                               {"Lab (PDF fonksiyonlarimiz)", bgr2lab_textbook, lab2bgr_textbook, cv::COLOR_BGR2Lab, cv::COLOR_Lab2BGR},
                               {"HSI (kendi fonksiyonlarimiz; OpenCV'de yok)", bgr2hsi, hsi2bgr, -1, -1}};
    printHeader("3) GIDIS-DONUS (BGR -> X -> BGR), orijinal resme gore hata");
    for (const RoundTrip& t : trips) {
        ConvFn f1 = t.fwd, f2 = t.inv;
        printCmp(std::string(t.name) + " [kendi]", compareMats(f2(f1(bgr)), bgr), pixels);
        if (t.cvForward >= 0) {
            cv::Mat mid, back;
            cv::cvtColor(bgr, mid, t.cvForward);
            cv::cvtColor(mid, back, t.cvInverse);
            printCmp(std::string("   ayni donusum, OpenCV ile [OpenCV]"), compareMats(back, bgr), pixels);
        }
    }
    std::cout << "\n";

    // =========================================================================================
    // 4) Herhangi bir uzaydan herhangi bir uzaya (BGR merkezli tablo, fonksiyon gostericileri)
    // =========================================================================================
    const char* spaceNames[4] = {"BGR", "YCrCb", "HSV", "Lab"};
    ConvFn toBgr[4] = {identityConv, ycrcb2bgr_exact, hsv2bgr_trunc, lab2bgr_textbook};
    ConvFn fromBgr[4] = {identityConv, bgr2ycrcb_exact, bgr2hsv_exact, bgr2lab_textbook};
    const int cvFwd[4] = {-1, cv::COLOR_BGR2YCrCb, cv::COLOR_BGR2HSV, cv::COLOR_BGR2Lab};
    const int cvInv[4] = {-1, cv::COLOR_YCrCb2BGR, cv::COLOR_HSV2BGR, cv::COLOR_Lab2BGR};
    printHeader("4) UZAYDAN UZAYA DONUSUM (tablodaki gostericilerle: once BGR'ye, sonra hedef uzaya) ve OpenCV ile iki adimli karsilastirma");
    const int pairs[][2] = {{1, 2}, {2, 1}, {1, 3}, {3, 2}};
    for (const auto& pr : pairs) {
        const int from = pr[0], to = pr[1];
        cv::Mat input = bgr, tmp;
        cv::cvtColor(bgr, input, cvFwd[from]);  // from uzayindaki girdi (OpenCV ile uretildi)
        const cv::Mat mine = fromBgr[to](toBgr[from](input));
        cv::cvtColor(input, tmp, cvInv[from]);
        cv::Mat ref;
        cv::cvtColor(tmp, ref, cvFwd[to]);
        printCmp(std::string(spaceNames[from]) + " -> " + spaceNames[to], compareMats(mine, ref), pixels);
    }
    std::cout << "\n";

    // =========================================================================================
    // 5) HSI ve normalize rgb: bilinen renklerle dogrulama
    // =========================================================================================
    std::cout << "5) HSI ve NORMALIZE RGB: bilinen renklerle dogrulama (R, G, B ile verilen tek piksel)\n";
    struct Probe {
        const char* name;
        int r, g, b;
        const char* expectedHsi;
        const char* expectedNrgb;
    };
    const Probe probes[] = {{"kirmizi", 255, 0, 0, "H = 0, S = 1, I = 0.333", "(1, 0, 0)"},         {"yesil", 0, 255, 0, "H = 120, S = 1, I = 0.333", "(0, 1, 0)"},
                            {"mavi", 0, 0, 255, "H = 240, S = 1, I = 0.333", "(0, 0, 1)"},         {"sari", 255, 255, 0, "H = 60, S = 1, I = 0.667", "(0.5, 0.5, 0)"},
                            {"beyaz", 255, 255, 255, "H = tanimsiz (0), S = 0, I = 1", "(1/3, 1/3, 1/3)"}, {"gri 128", 128, 128, 128, "H = tanimsiz (0), S = 0, I = 0.502", "(1/3, 1/3, 1/3)"},
                            {"siyah", 0, 0, 0, "H, S tanimsiz (0), I = 0", "tanimsiz (0, 0, 0)"}};
    cv::Mat pix(1, 1, CV_8UC3);
    std::cout << "  " << std::left << std::setw(10) << "Renk" << std::setw(34) << "HSI (hesaplanan)" << std::setw(40) << "HSI (beklenen)" << std::setw(24) << "norm. rgb (hesaplanan)" << "beklenen\n";
    for (const Probe& pr : probes) {
        pix.at<cv::Vec3b>(0, 0) = cv::Vec3b(static_cast<uchar>(pr.b), static_cast<uchar>(pr.g), static_cast<uchar>(pr.r));
        const cv::Vec3f h = bgr2hsi(pix).at<cv::Vec3f>(0, 0), n = bgr2nrgb(pix).at<cv::Vec3f>(0, 0);
        std::ostringstream a, b;
        a << std::fixed << std::setprecision(3) << "H = " << h[0] << ", S = " << h[1] << ", I = " << h[2];
        b << std::fixed << std::setprecision(3) << "(" << n[0] << ", " << n[1] << ", " << n[2] << ")";
        std::cout << "  " << std::left << std::setw(10) << pr.name << std::setw(34) << a.str() << std::setw(40) << pr.expectedHsi << std::setw(24) << b.str() << pr.expectedNrgb << "\n";
    }

    // gorsel: kanallar (kendi, OpenCV birebir fonksiyonlarimizla)
    {
        auto panel = [&](const cv::Mat& ch, const std::string& label) {
            cv::Mat s, bgr3;
            const double sc = 320.0 / ch.cols;
            cv::resize(ch, s, cv::Size(), sc, sc, cv::INTER_AREA);
            cv::cvtColor(s, bgr3, cv::COLOR_GRAY2BGR);
            cv::putText(bgr3, label, cv::Point(6, 18), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
            return bgr3;
        };
        std::vector<cv::Mat> rows;
        const struct {
            const char* n[3];
            cv::Mat img;
        } sets[3] = {{{"Y", "Cr", "Cb"}, bgr2ycrcb_exact(bgr)}, {{"H (x2 derece)", "S", "V"}, bgr2hsv_exact(bgr)}, {{"L", "a (+128)", "b (+128)"}, bgr2lab_textbook(bgr)}};
        for (const auto& s : sets) {
            std::vector<cv::Mat> ch, panels;
            cv::split(s.img, ch);
            for (int i = 0; i < 3; ++i) panels.push_back(panel(ch[i], s.n[i]));
            cv::Mat row;
            cv::hconcat(panels, row);
            rows.push_back(row);
        }
        cv::Mat all;
        cv::vconcat(rows, all);
        cv::imwrite((outDir / "4_kanallar_ycrcb_hsv_lab.png").string(), all);
    }
    std::cout << "\nKaydedilenler (" << outDir.string() << "): 4_kanallar_ycrcb_hsv_lab.png (ust: YCrCb kanallari, orta: HSV, alt: Lab)\n";
    return 0;
}

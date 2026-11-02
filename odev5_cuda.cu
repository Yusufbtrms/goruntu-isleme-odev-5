// Ödev 5 - Haftalık ödev: Görev 1-5'in CUDA ile yeniden gerçeklenmesi.
//
// Aynı piksel fonksiyonları hem CPU'da hem GPU'da çalışır (`__host__ __device__` işlevli yapılar): böylece CPU ve GPU sonuçları
// aynı mantıkla hesaplanır, karşılaştırma yalnızca paralel çalıştırmayı ve sayısal kütüphane farklarını sınar.
//
//   Görev 1-2: kanal histogramları (3 kanal): CPU, GPU naif (global atomik) ve GPU paylaşılan bellek (blok başına özel histogram)
//   Görev 3  : gri dönüşüm (15 bit sabit nokta)
//   Görev 4  : renk uzayı dönüşümleri: YCrCb, HSV, CIELAB (ileri ve ters), HSI (ileri ve ters), normalize rgb (ileri)
//   Görev 5  : X-ray görüntüsünü gri yapıp kendi paletimizle renklendirme (gri + arama tablosu; ayrıca tek çekirdekte birleşik)
//
// Her işlem için: CPU süresi (tek thread), GPU çekirdek süresi, GPU toplam süre (veriyi GPU'ya yükleme ve geri indirme dahil),
// hızlanma ve GPU sonucunun CPU sonucundan farkı (farklı eleman sayısı ve en büyük fark) yazdırılır.
//
// Derleme (Colab): nvcc -O2 -std=c++17 --fmad=false -arch=native odev5_cuda.cu -o odev5_cuda $(pkg-config --cflags --libs opencv4) -Xcompiler "-O2,-pthread"
// Kullanım: ./odev5_cuda [cikti_dizini] [renkli_resim=images/Cat.jpg] [xray_resim=images/ThreadsRGB/image_0_287.jpg] [en_uzun_kenar=0 (0 = kucultme yok)]

#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fsys = std::filesystem;

#define CUDA_CHECK(x)                                                                                    \
    do {                                                                                                 \
        cudaError_t e_ = (x);                                                                            \
        if (e_ != cudaSuccess) {                                                                         \
            std::cerr << "CUDA hatasi: " << cudaGetErrorString(e_) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            std::exit(1);                                                                                \
        }                                                                                                \
    } while (0)

#define KPI 3.14159265358979323846

// ---------------------------------------------------------------------------------------------
// Ortak yardımcılar (CPU ve GPU)
// ---------------------------------------------------------------------------------------------
__host__ __device__ inline int descale(int x, int n) { return (x + (1 << (n - 1))) >> n; }
__host__ __device__ inline unsigned char clamp8(int v) { return static_cast<unsigned char>(v < 0 ? 0 : (v > 255 ? 255 : v)); }
__host__ __device__ inline unsigned char sat8(double v) {
    const double r = rint(v);  // en yakin, esitlikte cifte yuvarlama (OpenCV'nin cvRound'u gibi)
    return static_cast<unsigned char>(r < 0.0 ? 0 : (r > 255.0 ? 255 : static_cast<int>(r)));
}
__host__ __device__ inline int imax3(int a, int b, int c) { return a > b ? (a > c ? a : c) : (b > c ? b : c); }
__host__ __device__ inline int imin3(int a, int b, int c) { return a < b ? (a < c ? a : c) : (b < c ? b : c); }

// HSV sabit nokta bolme tablolarinin tam sayi karsiligi (yuvarlama: esitlik olmadigi icin cvRound ile ayni)
__host__ __device__ inline int sdivf(int v) { return v > 0 ? (2 * 1044480 + v) / (2 * v) : 0; }  // round((255 << 12) / v)
__host__ __device__ inline int hdivf(int d) { return d > 0 ? (2 * 122880 + d) / (2 * d) : 0; }   // round((180 << 12) / (6 d))

__host__ __device__ inline double srgbToLinear(double c255) {
    const double c = c255 / 255.0;
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}
__host__ __device__ inline double linearToSrgb255(double l) {
    const double c = l <= 0.0031308 ? 12.92 * l : 1.055 * pow(l, 1.0 / 2.4) - 0.055;
    return c * 255.0;
}
__host__ __device__ inline double labF(double t) { return t > 216.0 / 24389.0 ? cbrt(t) : t / (108.0 / 841.0) + 4.0 / 29.0; }
__host__ __device__ inline double labFinv(double t) {
    const double d = 6.0 / 29.0;
    return t > d ? t * t * t : (108.0 / 841.0) * (t - 4.0 / 29.0);
}

// ---------------------------------------------------------------------------------------------
// Piksel işlemleri (functor): In -> Out, hem CPU hem GPU
// ---------------------------------------------------------------------------------------------
struct GrayOp {
    using In = uchar3;
    using Out = unsigned char;
    __host__ __device__ void operator()(const uchar3& p, unsigned char& o) const { o = static_cast<unsigned char>((p.z * 9798 + p.y * 19235 + p.x * 3735 + (1 << 14)) >> 15); }
};

struct YCrCbOp {  // BGR -> (Y, Cr, Cb), OpenCV birebir
    using In = uchar3;
    using Out = uchar3;
    __host__ __device__ void operator()(const uchar3& p, uchar3& o) const {
        const int B = p.x, G = p.y, R = p.z;
        const int Y = descale(R * 4899 + G * 9617 + B * 1868, 14);
        const int Cr = descale((R - Y) * 11682 + (128 << 14), 14);
        const int Cb = descale((B - Y) * 9241 + (128 << 14), 14);
        o = make_uchar3(clamp8(Y), clamp8(Cr), clamp8(Cb));
    }
};

struct YCrCbInvOp {  // (Y, Cr, Cb) -> BGR, OpenCV birebir
    using In = uchar3;
    using Out = uchar3;
    __host__ __device__ void operator()(const uchar3& p, uchar3& o) const {
        const int Y = p.x, Cr = p.y - 128, Cb = p.z - 128;
        const int b = Y + descale(Cb * 29049, 14);
        const int g = Y + descale(Cb * (-5636) + Cr * (-11698), 14);
        const int r = Y + descale(Cr * 22987, 14);
        o = make_uchar3(clamp8(b), clamp8(g), clamp8(r));
    }
};

struct HsvOp {  // BGR -> (H/2, S, V), OpenCV birebir
    using In = uchar3;
    using Out = uchar3;
    __host__ __device__ void operator()(const uchar3& p, uchar3& o) const {
        const int b = p.x, g = p.y, r = p.z;
        const int v = imax3(b, g, r), vmin = imin3(b, g, r), diff = v - vmin;
        const int vr = (v == r) ? -1 : 0, vg = (v == g) ? -1 : 0;
        const int s = (diff * sdivf(v) + (1 << 11)) >> 12;
        int h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
        h = (h * hdivf(diff) + (1 << 11)) >> 12;
        h += h < 0 ? 180 : 0;
        o = make_uchar3(clamp8(h), clamp8(s), clamp8(v));
    }
};

struct HsvInvOp {  // (H/2, S, V) -> BGR, float, kesmeli (OpenCV ile uyumlu)
    using In = uchar3;
    using Out = uchar3;
    __host__ __device__ static unsigned char toU8(float v) { return clamp8(static_cast<int>(v * 255.f)); }
    __host__ __device__ void operator()(const uchar3& p, uchar3& o) const {
        float h = p.x, s = p.y * (1.f / 255.f), v = p.z * (1.f / 255.f);
        float b, g, r;
        if (s == 0) {
            b = g = r = v;
        } else {
            h *= 6.f / 180.f;
            while (h < 0) h += 6;
            while (h >= 6) h -= 6;
            int sector = static_cast<int>(floorf(h));
            h -= sector;
            if (static_cast<unsigned>(sector) >= 6u) {
                sector = 0;
                h = 0.f;
            }
            const float t0 = v, t1 = v * (1.f - s), t2 = v * (1.f - s * h), t3 = v * (1.f - s * (1.f - h));
            switch (sector) {
                case 0: b = t1; g = t3; r = t0; break;
                case 1: b = t1; g = t0; r = t2; break;
                case 2: b = t3; g = t0; r = t1; break;
                case 3: b = t0; g = t2; r = t1; break;
                case 4: b = t0; g = t1; r = t3; break;
                default: b = t2; g = t1; r = t0; break;
            }
        }
        o = make_uchar3(toU8(b), toU8(g), toU8(r));
    }
};

struct LabOp {  // BGR -> Lab (PDF formulu; 8-bit: L*255/100, a+128, b+128)
    using In = uchar3;
    using Out = uchar3;
    __host__ __device__ void operator()(const uchar3& p, uchar3& o) const {
        const double Xn = 0.95047, Yn = 1.0, Zn = 1.08883;
        const double R = srgbToLinear(p.z), G = srgbToLinear(p.y), B = srgbToLinear(p.x);
        const double X = 0.4124 * R + 0.3576 * G + 0.1805 * B;
        const double Y = 0.2126 * R + 0.7152 * G + 0.0722 * B;
        const double Z = 0.0193 * R + 0.1192 * G + 0.9505 * B;
        const double fx = labF(X / Xn), fy = labF(Y / Yn), fz = labF(Z / Zn);
        const double L = 116.0 * fy - 16.0, a = 500.0 * (fx - fy), b = 200.0 * (fy - fz);
        o = make_uchar3(sat8(L * 255.0 / 100.0), sat8(a + 128.0), sat8(b + 128.0));
    }
};

struct LabInvOp {  // Lab -> BGR; m = XYZ->dogrusal RGB matrisinin tersi (satir sirasiyla)
    using In = uchar3;
    using Out = uchar3;
    double m[9];
    __host__ __device__ void operator()(const uchar3& p, uchar3& o) const {
        const double L = p.x * 100.0 / 255.0, a = p.y - 128.0, b = p.z - 128.0;
        const double fy = (L + 16.0) / 116.0, fx = fy + a / 500.0, fz = fy - b / 200.0;
        const double X = 0.95047 * labFinv(fx), Y = 1.0 * labFinv(fy), Z = 1.08883 * labFinv(fz);
        const double r = m[0] * X + m[1] * Y + m[2] * Z, g = m[3] * X + m[4] * Y + m[5] * Z, bl = m[6] * X + m[7] * Y + m[8] * Z;
        o = make_uchar3(sat8(linearToSrgb255(fmax(0.0, bl))), sat8(linearToSrgb255(fmax(0.0, g))), sat8(linearToSrgb255(fmax(0.0, r))));
    }
};

struct HsiOp {  // BGR -> (H derece, S, I), float
    using In = uchar3;
    using Out = float3;
    __host__ __device__ void operator()(const uchar3& p, float3& o) const {
        const double R = p.z / 255.0, G = p.y / 255.0, B = p.x / 255.0;
        const double sum = R + G + B, I = sum / 3.0;
        const double mn = R < G ? (R < B ? R : B) : (G < B ? G : B);
        const double S = sum > 0.0 ? 1.0 - 3.0 * mn / sum : 0.0;
        const double num = 0.5 * ((R - G) + (R - B));
        const double den = sqrt((R - G) * (R - G) + (R - B) * (G - B));
        double H = 0.0;
        if (den > 1e-12) {
            double c = num / den;
            c = c < -1.0 ? -1.0 : (c > 1.0 ? 1.0 : c);
            const double theta = acos(c) * 180.0 / KPI;
            H = (B <= G) ? theta : 360.0 - theta;
        }
        o = make_float3(static_cast<float>(H), static_cast<float>(S), static_cast<float>(I));
    }
};

struct HsiInvOp {  // (H, S, I) -> BGR
    using In = float3;
    using Out = uchar3;
    __host__ __device__ void operator()(const float3& p, uchar3& o) const {
        const double H = p.x, S = p.y, I = p.z, rad = KPI / 180.0;
        double R, G, B;
        if (H < 120.0) {
            B = I * (1.0 - S);
            R = I * (1.0 + S * cos(H * rad) / cos((60.0 - H) * rad));
            G = 3.0 * I - (R + B);
        } else if (H < 240.0) {
            const double h = H - 120.0;
            R = I * (1.0 - S);
            G = I * (1.0 + S * cos(h * rad) / cos((60.0 - h) * rad));
            B = 3.0 * I - (R + G);
        } else {
            const double h = H - 240.0;
            G = I * (1.0 - S);
            B = I * (1.0 + S * cos(h * rad) / cos((60.0 - h) * rad));
            R = 3.0 * I - (G + B);
        }
        o = make_uchar3(sat8(B * 255.0), sat8(G * 255.0), sat8(R * 255.0));
    }
};

struct NrgbOp {  // BGR -> (r, g, b) normalize rgb, float
    using In = uchar3;
    using Out = float3;
    __host__ __device__ void operator()(const uchar3& p, float3& o) const {
        const float R = p.z, G = p.y, B = p.x, S = R + G + B;
        o = S > 0.f ? make_float3(R / S, G / S, B / S) : make_float3(0.f, 0.f, 0.f);
    }
};

struct PaletteOp {  // gri -> renk (arama tablosu)
    using In = unsigned char;
    using Out = uchar3;
    const uchar3* lut;
    __host__ __device__ void operator()(const unsigned char& g, uchar3& o) const { o = lut[g]; }
};

struct XrayOp {  // BGR -> gri -> renk (tek cekirdekte birlesik)
    using In = uchar3;
    using Out = uchar3;
    const uchar3* lut;
    __host__ __device__ void operator()(const uchar3& p, uchar3& o) const {
        const int g = (p.z * 9798 + p.y * 19235 + p.x * 3735 + (1 << 14)) >> 15;
        o = lut[g];
    }
};

// ---------------------------------------------------------------------------------------------
// GPU çekirdekleri
// ---------------------------------------------------------------------------------------------
template <typename Op>
__global__ void mapKernel(const typename Op::In* in, typename Op::Out* out, int n, Op op) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) op(in[i], out[i]);
}

__global__ void histNaive(const uchar3* img, int n, unsigned int* hist) {  // global bellekte atomik toplama
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const uchar3 p = img[i];
        atomicAdd(&hist[p.x], 1u);
        atomicAdd(&hist[256 + p.y], 1u);
        atomicAdd(&hist[512 + p.z], 1u);
    }
}

__global__ void histShared(const uchar3* img, int n, unsigned int* hist) {  // blok basina ozel histogram (paylasilan bellek)
    __shared__ unsigned int sh[768];
    for (int t = threadIdx.x; t < 768; t += blockDim.x) sh[t] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const uchar3 p = img[i];
        atomicAdd(&sh[p.x], 1u);
        atomicAdd(&sh[256 + p.y], 1u);
        atomicAdd(&sh[512 + p.z], 1u);
    }
    __syncthreads();
    for (int t = threadIdx.x; t < 768; t += blockDim.x) atomicAdd(&hist[t], sh[t]);
}

// ---------------------------------------------------------------------------------------------
// Ölçüm ve karşılaştırma
// ---------------------------------------------------------------------------------------------
static double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static void cmpElem(unsigned char a, unsigned char b, double tol, double& sum, double& mx, bool& diff) {
    const double d = std::fabs(static_cast<double>(a) - static_cast<double>(b));
    sum += d;
    mx = std::max(mx, d);
    if (d > tol) diff = true;
}
static void cmpElem(const uchar3& a, const uchar3& b, double tol, double& sum, double& mx, bool& diff) {
    cmpElem(a.x, b.x, tol, sum, mx, diff);
    cmpElem(a.y, b.y, tol, sum, mx, diff);
    cmpElem(a.z, b.z, tol, sum, mx, diff);
}
static void cmpElem(const float3& a, const float3& b, double tol, double& sum, double& mx, bool& diff) {
    const double d[3] = {std::fabs(static_cast<double>(a.x) - b.x), std::fabs(static_cast<double>(a.y) - b.y), std::fabs(static_cast<double>(a.z) - b.z)};
    for (double v : d) {
        sum += v;
        mx = std::max(mx, v);
        if (!(v <= tol)) diff = true;
    }
}

template <typename Out>
struct Result {
    std::vector<Out> cpu, gpu;
};

static void printHeader(const std::string& title) {
    std::cout << title << "\n  " << std::left << std::setw(40) << "Islem" << std::right << std::setw(11) << "CPU ms" << std::setw(13) << "GPU cekirdek" << std::setw(11) << "GPU toplam" << std::setw(13) << "hizl. cek." << std::setw(13)
              << "hizl. toplam" << std::setw(14) << "farkli eleman" << std::setw(11) << "en buyuk" << "\n";
}

static void printRow(const std::string& name, double cpuMs, double kMs, double tMs, long long diffCount, double maxDiff) {
    std::cout << "  " << std::left << std::setw(40) << name << std::right << std::fixed << std::setprecision(3) << std::setw(11) << cpuMs << std::setw(13) << kMs << std::setw(11) << tMs << std::setprecision(1) << std::setw(12)
              << cpuMs / kMs << "x" << std::setw(12) << cpuMs / tMs << "x" << std::setw(14) << diffCount << std::setprecision(6) << std::setw(11) << maxDiff << "\n";
}

template <typename Op>
Result<typename Op::Out> runOp(const std::string& name, const Op& opCpu, const Op& opGpu, const std::vector<typename Op::In>& in, double tol = 0.0) {
    using In = typename Op::In;
    using Out = typename Op::Out;
    const int n = static_cast<int>(in.size());
    Result<Out> res;
    res.cpu.resize(n);
    res.gpu.resize(n);
    const int cpuReps = 3, gpuReps = 30, totalReps = 10;

    double t0 = nowMs();
    for (int r = 0; r < cpuReps; ++r)
        for (int i = 0; i < n; ++i) opCpu(in[i], res.cpu[i]);
    const double cpuMs = (nowMs() - t0) / cpuReps;

    In* dIn;
    Out* dOut;
    CUDA_CHECK(cudaMalloc(&dIn, sizeof(In) * n));
    CUDA_CHECK(cudaMalloc(&dOut, sizeof(Out) * n));
    const int threads = 256, blocks = std::min((n + threads - 1) / threads, 4096);
    CUDA_CHECK(cudaMemcpy(dIn, in.data(), sizeof(In) * n, cudaMemcpyHostToDevice));
    mapKernel<Op><<<blocks, threads>>>(dIn, dOut, n, opGpu);  // isinma
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int r = 0; r < gpuReps; ++r) mapKernel<Op><<<blocks, threads>>>(dIn, dOut, n, opGpu);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    const double kMs = ms / gpuReps;

    t0 = nowMs();
    for (int r = 0; r < totalReps; ++r) {
        CUDA_CHECK(cudaMemcpy(dIn, in.data(), sizeof(In) * n, cudaMemcpyHostToDevice));
        mapKernel<Op><<<blocks, threads>>>(dIn, dOut, n, opGpu);
        CUDA_CHECK(cudaMemcpy(res.gpu.data(), dOut, sizeof(Out) * n, cudaMemcpyDeviceToHost));
    }
    const double tMs = (nowMs() - t0) / totalReps;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));

    long long differing = 0;
    double sum = 0.0, mx = 0.0;
    for (int i = 0; i < n; ++i) {
        bool diff = false;
        cmpElem(res.cpu[i], res.gpu[i], tol, sum, mx, diff);
        if (diff) ++differing;
    }
    printRow(name, cpuMs, kMs, tMs, differing, mx);
    return res;
}

static std::vector<uchar3> toVec(const cv::Mat& m) {
    CV_Assert(m.type() == CV_8UC3 && m.isContinuous());
    std::vector<uchar3> v(m.total());
    std::memcpy(v.data(), m.data, m.total() * 3);
    return v;
}

static std::vector<uchar3> makeOwnPalette() {
    struct Anchor {
        double r, g, b;
    };
    const Anchor anchors[] = {{15, 45, 160}, {30, 125, 235}, {130, 165, 30}, {240, 140, 30}, {255, 247, 190}};
    struct Stop {
        double y, cb, cr;
    };
    std::vector<Stop> st;
    st.push_back({0.0, 128.0, 128.0});
    for (const Anchor& a : anchors)
        st.push_back({0.299 * a.r + 0.587 * a.g + 0.114 * a.b, 128.0 - 0.168736 * a.r - 0.331264 * a.g + 0.5 * a.b, 128.0 + 0.5 * a.r - 0.418688 * a.g - 0.081312 * a.b});
    st.push_back({255.0, 128.0, 128.0});
    std::vector<uchar3> lut(256);
    for (int g = 0; g < 256; ++g) {
        size_t i = 0;
        while (i + 2 < st.size() && g > st[i + 1].y) ++i;
        const double t = (g - st[i].y) / (st[i + 1].y - st[i].y);
        const double cb = st[i].cb + t * (st[i + 1].cb - st[i].cb), cr = st[i].cr + t * (st[i + 1].cr - st[i].cr);
        const double R = g + 1.402 * (cr - 128.0), G = g - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0), B = g + 1.772 * (cb - 128.0);
        lut[g] = make_uchar3(sat8(B), sat8(G), sat8(R));
    }
    return lut;
}

static cv::Mat loadImage(const std::string& path, int maxSide) {
    cv::Mat m = cv::imread(path, cv::IMREAD_COLOR);
    if (m.empty()) return m;
    if (maxSide > 0 && std::max(m.cols, m.rows) > maxSide) {
        const double s = static_cast<double>(maxSide) / std::max(m.cols, m.rows);
        cv::resize(m, m, cv::Size(), s, s, cv::INTER_AREA);
    }
    return m;
}

int main(int argc, char** argv) {
    const fsys::path outDir = (argc >= 2) ? fsys::path(argv[1]) : fsys::path("cikti_cuda");
    const std::string colorPath = (argc >= 3) ? argv[2] : "images/Cat.jpg";
    const std::string xrayPath = (argc >= 4) ? argv[3] : "images/ThreadsRGB/image_0_287.jpg";
    const int maxSide = (argc >= 5) ? std::stoi(argv[4]) : 0;
    fsys::create_directories(outDir);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (hesaplama yetenegi " << prop.major << "." << prop.minor << ", " << prop.multiProcessorCount << " islemci)\n";

    const cv::Mat color = loadImage(colorPath, maxSide);
    if (color.empty()) {
        std::cerr << "Hata: resim okunamadi: " << colorPath << "\n";
        return 1;
    }
    const std::vector<uchar3> colorVec = toVec(color);
    const int n = static_cast<int>(colorVec.size());
    std::cout << "Renkli resim: " << colorPath << " (" << color.cols << "x" << color.rows << ", " << n << " piksel)\n\n";

    // =========================================================================================
    // Gorev 1-2: histogram
    // =========================================================================================
    {
        std::cout << "GOREV 1-2: KANAL HISTOGRAMLARI (3 kanal x 256 kutu)\n";
        std::cout << "  " << std::left << std::setw(40) << "Yontem" << std::right << std::setw(11) << "CPU ms" << std::setw(13) << "GPU cekirdek" << std::setw(11) << "GPU toplam" << std::setw(13) << "hizl. cek." << std::setw(15)
                  << "|GPU - CPU| top." << "\n";
        std::vector<unsigned int> cpuHist(768, 0);
        double t0 = nowMs();
        for (int r = 0; r < 3; ++r) {
            std::fill(cpuHist.begin(), cpuHist.end(), 0u);
            for (const uchar3& p : colorVec) {
                ++cpuHist[p.x];
                ++cpuHist[256 + p.y];
                ++cpuHist[512 + p.z];
            }
        }
        const double cpuMs = (nowMs() - t0) / 3;

        uchar3* dImg;
        unsigned int* dHist;
        CUDA_CHECK(cudaMalloc(&dImg, sizeof(uchar3) * n));
        CUDA_CHECK(cudaMalloc(&dHist, 768 * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(dImg, colorVec.data(), sizeof(uchar3) * n, cudaMemcpyHostToDevice));
        const int threads = 256, blocks = std::min((n + threads - 1) / threads, 2048);
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        for (int variant = 0; variant < 2; ++variant) {
            auto launch = [&]() {
                CUDA_CHECK(cudaMemset(dHist, 0, 768 * sizeof(unsigned int)));
                if (variant == 0) histNaive<<<blocks, threads>>>(dImg, n, dHist);
                else histShared<<<blocks, threads>>>(dImg, n, dHist);
            };
            launch();
            CUDA_CHECK(cudaDeviceSynchronize());  // isinma
            CUDA_CHECK(cudaEventRecord(e0));
            for (int r = 0; r < 30; ++r) launch();
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            const double kMs = ms / 30;
            std::vector<unsigned int> gpuHist(768);
            t0 = nowMs();
            for (int r = 0; r < 10; ++r) {
                CUDA_CHECK(cudaMemcpy(dImg, colorVec.data(), sizeof(uchar3) * n, cudaMemcpyHostToDevice));
                launch();
                CUDA_CHECK(cudaMemcpy(gpuHist.data(), dHist, 768 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            }
            const double tMs = (nowMs() - t0) / 10;
            long long diff = 0;
            for (int i = 0; i < 768; ++i) diff += std::llabs(static_cast<long long>(gpuHist[i]) - static_cast<long long>(cpuHist[i]));
            std::cout << "  " << std::left << std::setw(40) << (variant == 0 ? "GPU naif (global atomik)" : "GPU paylasilan bellek (blok basina)") << std::right << std::fixed << std::setprecision(3) << std::setw(11) << cpuMs
                      << std::setw(13) << kMs << std::setw(11) << tMs << std::setprecision(1) << std::setw(12) << cpuMs / kMs << "x" << std::setw(15) << diff << "\n";
        }
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaFree(dImg));
        CUDA_CHECK(cudaFree(dHist));
        std::cout << "\n";
    }

    // =========================================================================================
    // Gorev 3 ve 4: gri ve renk uzayi donusumleri
    // =========================================================================================
    printHeader("GOREV 3 ve 4: GRI VE RENK UZAYI DONUSUMLERI (renkli resim; farkli eleman: kanal bazinda en az bir farkli deger olan piksel sayisi)");
    const Result<unsigned char> gray = runOp<GrayOp>("Gri (15 bit sabit nokta)", GrayOp{}, GrayOp{}, colorVec);
    const Result<uchar3> ycc = runOp<YCrCbOp>("BGR -> YCrCb", YCrCbOp{}, YCrCbOp{}, colorVec);
    runOp<YCrCbInvOp>("YCrCb -> BGR", YCrCbInvOp{}, YCrCbInvOp{}, ycc.cpu);
    const Result<uchar3> hsv = runOp<HsvOp>("BGR -> HSV", HsvOp{}, HsvOp{}, colorVec);
    runOp<HsvInvOp>("HSV -> BGR (kesmeli)", HsvInvOp{}, HsvInvOp{}, hsv.cpu);
    const Result<uchar3> lab = runOp<LabOp>("BGR -> Lab (PDF formulu)", LabOp{}, LabOp{}, colorVec);
    {
        const cv::Matx33d M(0.4124, 0.3576, 0.1805, 0.2126, 0.7152, 0.0722, 0.0193, 0.1192, 0.9505);
        const cv::Matx33d Mi = M.inv();
        LabInvOp op;
        for (int i = 0; i < 9; ++i) op.m[i] = Mi(i / 3, i % 3);
        runOp<LabInvOp>("Lab -> BGR (PDF formulu)", op, op, lab.cpu);
    }
    const Result<float3> hsi = runOp<HsiOp>("BGR -> HSI (float; esik 1e-3)", HsiOp{}, HsiOp{}, colorVec, 1e-3);
    runOp<HsiInvOp>("HSI -> BGR", HsiInvOp{}, HsiInvOp{}, hsi.cpu);
    runOp<NrgbOp>("BGR -> normalize rgb (float; esik 1e-6)", NrgbOp{}, NrgbOp{}, colorVec, 1e-6);

    // bilgi: OpenCV'nin kendi cvtColor'i ile (bu makinedeki OpenCV surumune bagli)
    {
        auto info = [&](const char* name, const cv::Mat& mine, int code) {
            cv::Mat ref, d;
            cv::cvtColor(color, ref, code);
            cv::absdiff(mine, ref, d);
            d = d.reshape(1);
            std::cout << "  " << std::left << std::setw(26) << name << "farkli deger sayisi: " << cv::countNonZero(d) << "  (mutlak fark toplami " << static_cast<long long>(cv::norm(d, cv::NORM_L1)) << ")\n";
        };
        std::cout << "\n  Bilgi: kendi (CPU) sonuclarimiz ile bu makinedeki OpenCV " << CV_VERSION << " cvtColor sonucu (surum farki varsa sifir olmayabilir):\n";
        info("Gri", cv::Mat(color.rows, color.cols, CV_8UC1, const_cast<unsigned char*>(gray.cpu.data())), cv::COLOR_BGR2GRAY);
        info("YCrCb", cv::Mat(color.rows, color.cols, CV_8UC3, const_cast<uchar3*>(ycc.cpu.data())), cv::COLOR_BGR2YCrCb);
        info("HSV", cv::Mat(color.rows, color.cols, CV_8UC3, const_cast<uchar3*>(hsv.cpu.data())), cv::COLOR_BGR2HSV);
        info("Lab (PDF formulu)", cv::Mat(color.rows, color.cols, CV_8UC3, const_cast<uchar3*>(lab.cpu.data())), cv::COLOR_BGR2Lab);
    }

    // =========================================================================================
    // Gorev 5: X-ray gri + renklendirme
    // =========================================================================================
    std::cout << "\n";
    const cv::Mat xray = loadImage(xrayPath, maxSide);
    if (xray.empty()) {
        std::cout << "GOREV 5 atlandi: X-ray resmi okunamadi: " << xrayPath << "\n";
    } else {
        const std::vector<uchar3> xVec = toVec(xray);
        std::cout << "X-ray resmi: " << xrayPath << " (" << xray.cols << "x" << xray.rows << ", " << xVec.size() << " piksel)\n";
        printHeader("GOREV 5: X-RAY GRI VE RENKLENDIRME");
        const std::vector<uchar3> lutHost = makeOwnPalette();
        uchar3* dLut;
        CUDA_CHECK(cudaMalloc(&dLut, 256 * sizeof(uchar3)));
        CUDA_CHECK(cudaMemcpy(dLut, lutHost.data(), 256 * sizeof(uchar3), cudaMemcpyHostToDevice));
        const Result<unsigned char> xg = runOp<GrayOp>("X-ray: gri", GrayOp{}, GrayOp{}, xVec);
        const Result<uchar3> xc = runOp<PaletteOp>("X-ray: gri -> renk (arama tablosu)", PaletteOp{lutHost.data()}, PaletteOp{dLut}, xg.cpu);
        const Result<uchar3> xf = runOp<XrayOp>("X-ray: BGR -> renk (tek cekirdekte)", XrayOp{lutHost.data()}, XrayOp{dLut}, xVec);
        CUDA_CHECK(cudaFree(dLut));
        cv::imwrite((outDir / "cuda_xray_gri.png").string(), cv::Mat(xray.rows, xray.cols, CV_8UC1, const_cast<unsigned char*>(xg.gpu.data())));
        cv::imwrite((outDir / "cuda_xray_renkli.png").string(), cv::Mat(xray.rows, xray.cols, CV_8UC3, const_cast<uchar3*>(xf.gpu.data())));
        std::cout << "  GPU sonuclari kaydedildi: " << (outDir / "cuda_xray_gri.png").string() << ", " << (outDir / "cuda_xray_renkli.png").string() << "\n";
        (void)xc;
    }
    return 0;
}

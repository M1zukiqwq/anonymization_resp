// mandel perf/correctness source for validating the contiguous-scatter opt.
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdio>
#include <mpi.h>
#include "ReconTensor.h"
using namespace std;
namespace dacpp { typedef std::vector<std::any> list; }

const int row_count = 8192, col_count = 8192, max_iterations = 1000;
vector<complex<float>> complex_points;
vector<int> mandelbrot_flags;
int total_points = 0;
long long mandelbrot_count = 0;

void InitializeComplexPoints() {
    total_points = row_count * col_count;
    complex_points.resize(total_points);
    for (int i = 0; i < row_count; ++i)
        for (int j = 0; j < col_count; ++j) {
            int index = i * col_count + j;
            float real = -1.5f + (i * (2.0f / row_count));
            float imag = -1.0f + (j * (2.0f / col_count));
            complex_points[index] = complex<float>(real, imag);
        }
}

shell dacpp::list MANDEL(const dacpp::Vector<complex<float>>& complex_points,
                         dacpp::Vector<int>& mandelbrot_flags) {
    dacpp::index i;
    dacpp::list dataList{complex_points[i], mandelbrot_flags[i]};
    return dataList;
}

calc void mandel(complex<float>* complex_points, int* mandelbrot_flags) {
    const complex<float>& c = complex_points[0];
    complex<float> z = 0;
    int iterations = 0;
    for (int i = 0; i < max_iterations; ++i) {
        if (std::sqrt(z.real() * z.real() + z.imag() * z.imag()) > 2.0f) { iterations = i; break; }
        z = z * z + c;
        iterations = max_iterations;
    }
    if (iterations == max_iterations) mandelbrot_flags[0] = 1;
}

static void __dacpp_e2e(const char* name, double secs) {
    double mx = 0.0, sm = 0.0;
    MPI_Reduce(&secs, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&secs, &sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    int r = 0, s = 1; MPI_Comm_rank(MPI_COMM_WORLD, &r); MPI_Comm_size(MPI_COMM_WORLD, &s);
    if (r == 0) std::printf("[MPI TEST] %s | ranks=%d | e2e_max=%.6f s | e2e_avg=%.6f s\n", name, s, mx, sm / s);
}

int main() {
    InitializeComplexPoints();
    mandelbrot_flags.resize(total_points, 0);
    dacpp::Vector<complex<float>> complex_points_tensor(complex_points);
    dacpp::Vector<int> mandelbrot_flags_tensor(mandelbrot_flags);

    double __t0 = MPI_Wtime();
    MANDEL(complex_points_tensor, mandelbrot_flags_tensor) <-> mandel;
    double __el = MPI_Wtime() - __t0;

    mandelbrot_flags_tensor.tensor2Array(mandelbrot_flags);
    long long cnt = 0; for (int f : mandelbrot_flags) cnt += f;
    int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r);
    if (r == 0) std::printf("[CHECK] mandel bounded_count=%lld total=%d\n", cnt, total_points);
    __dacpp_e2e("mandel.opt", __el);
    return 0;
}

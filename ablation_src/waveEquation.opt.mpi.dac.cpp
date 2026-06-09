// waveEquation perf/correctness source (1024^2, 500 steps).
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <mpi.h>
#include "ReconTensor.h"
using namespace std;
namespace dacpp { typedef std::vector<std::any> list; }

const int NX = 1024;
const int NY = 1024;
const double Lx = 10.0, Ly = 10.0, c = 1.0;
const int TIME_STEPS = 500;
const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);
const double dt = 0.5 * std::fmin(dx, dy) / c;

shell dacpp::list waveEqShell(dacpp::Matrix<double>& matCur WRITE,
                              dacpp::Matrix<double>& matPrev READ_WRITE,
                              dacpp::Matrix<double>& matNext WRITE) {
    dacpp::split sp1(3, 1), sp2(3, 1);
    dacpp::index idx1, idx2;
    binding(sp1, idx1);
    binding(sp2, idx2);
    dacpp::list dataList{matCur[sp1][sp2], matPrev[idx1][idx2], matNext[idx1][idx2]};
    return dataList;
}

calc void waveEq(dacpp::Matrix<double>& cur, double* prev, double* next) {
    double dt = 0.5 * std::fmin(dx, dy) / c;
    double u_xx = (cur[2][1] - 2.0 * cur[1][1] + cur[0][1]) / (dx * dx);
    double u_yy = (cur[1][2] - 2.0 * cur[1][1] + cur[1][0]) / (dy * dy);
    next[0] = 2.0 * cur[1][1] - prev[0] + (c * c) * dt * dt * (u_xx + u_yy);
}

static void __dacpp_e2e(const char* name, double secs) {
    double mx = 0.0, sm = 0.0;
    MPI_Reduce(&secs, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&secs, &sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    int r = 0, s = 1; MPI_Comm_rank(MPI_COMM_WORLD, &r); MPI_Comm_size(MPI_COMM_WORLD, &s);
    if (r == 0) std::printf("[MPI TEST] %s | ranks=%d | e2e_max=%.6f s | e2e_avg=%.6f s\n", name, s, mx, sm / s);
}

int main() {
    vector<double> u_prev(NX * NY, 0.0), u_curr(NX * NY, 0.0), u_next(NX * NY, 0.0);
    double sigma = 0.5;
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j) {
            double x = i * dx, y = j * dy;
            u_prev[i * NX + j] = std::exp(-((x - Lx / 2) * (x - Lx / 2) + (y - Ly / 2) * (y - Ly / 2)) / (2 * sigma * sigma));
        }
    dacpp::Matrix<double> matCur({NX, NY}, u_curr);
    dacpp::Matrix<double> u_prev_tensor({NX, NY}, u_prev);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> matPrev = u_prev_tensor[{1, NX - 1}][{1, NY - 1}];
    dacpp::Matrix<double> matNext = u_next_tensor[{1, NX - 1}][{1, NY - 1}];

    double __t0 = MPI_Wtime();
    for (int t = 0; t < TIME_STEPS; t++) {
        waveEqShell(matCur, matPrev, matNext) <-> waveEq;
        for (int i = 1; i <= NX - 2; i++) for (int j = 1; j <= NY - 2; j++) matPrev[i - 1][j - 1] = matCur[i][j];
        for (int i = 1; i <= NX - 2; i++) for (int j = 1; j <= NY - 2; j++) matCur[i][j] = matNext[i - 1][j - 1];
        for (int i = 0; i <= NX - 1; ++i) { matCur[i][NY - 1] = 0; matCur[i][0] = 0; }
        for (int j = 0; j <= NY - 1; ++j) { matCur[NX - 1][j] = 0; matCur[0][j] = 0; }
    }
    double __el = MPI_Wtime() - __t0;

    int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r);
    if (r == 0) {
        double cs = 0.0; for (int j = 0; j < NY; ++j) cs += matCur[NX / 2][j];
        std::printf("[CHECK] wave midrow_checksum=%.6f\n", cs);
    }
    __dacpp_e2e("wave.opt", __el);
    return 0;
}

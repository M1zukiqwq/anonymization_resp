// stencil ablation benchmark source (DSL + e2e timing harness).
// Translate ON:  --mpi                    (StencilWindow2D + resident-halo)
// Translate OFF: --mpi --mpi-no-resident-halo  (StencilFullSync)
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <mpi.h>
#include "ReconTensor.h"

using namespace std;
namespace dacpp {
    typedef std::vector<std::any> list;
}

const int NX = 1024;
const int NY = 1024;
const double Lx = 10.0;
const double Ly = 10.0;
const double alpha = 0.01;
const int TIME_STEPS = 500;

const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);
const double dt_stability = (dx * dx * dy * dy) / (2.0 * alpha * (dx * dx + dy * dy));
const double delta_t = 0.4 * dt_stability;

shell dacpp::list stencilShell(dacpp::Matrix<double>& matIn READ_WRITE,
                               dacpp::Matrix<double>& matOut READ_WRITE) {
    dacpp::split sp1(3, 1), sp2(3, 1);
    dacpp::index idx1, idx2;
    binding(sp1, idx1);
    binding(sp2, idx2);
    dacpp::list dataList{matIn[sp1][sp2], matOut[idx1][idx2]};
    return dataList;
}

calc void stencil(dacpp::Matrix<double>& mat, double* out) {
    out[0] = mat[1][1] + alpha * delta_t * (((mat[2][1] - 2.0 * mat[1][1] + mat[0][1]) / (dx * dx)) + ((mat[1][2] - 2.0 * mat[1][1] + mat[1][0]) / (dy * dy)));
}

static void __dacpp_e2e(const char* name, double secs) {
    double mx = 0.0, sm = 0.0;
    MPI_Reduce(&secs, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&secs, &sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    int r = 0, s = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    if (r == 0)
        std::printf("[MPI TEST] %s | ranks=%d | e2e_max=%.6f s | e2e_avg=%.6f s\n",
                    name, s, mx, sm / s);
}

int main() {
    vector<double> u_curr(NX * NY, 0.0);
    vector<double> u_next(NX * NY, 0.0);
    double sigma = 1.0;
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j) {
            double x = i * dx, y = j * dy;
            u_curr[i * NY + j] = std::exp(-((x - Lx / 2.0) * (x - Lx / 2.0) +
                                            (y - Ly / 2.0) * (y - Ly / 2.0)) /
                                          (2.0 * sigma * sigma));
        }

    dacpp::Matrix<double> matIn({NX, NY}, u_curr);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> matOut = u_next_tensor[{1, NX - 1}][{1, NY - 1}];

    double __t0 = MPI_Wtime();
    for (int i = 0; i < TIME_STEPS; i++) {
        stencilShell(matIn, matOut) <-> stencil;
        for (int a = 1; a <= NX - 2; a++)
            for (int b = 1; b <= NY - 2; b++)
                matIn[a][b] = matOut[a - 1][b - 1];
        for (int j = 0; j <= NY - 1; ++j) {
            matIn[0][j] = matIn[1][j];
            matIn[NX - 1][j] = matIn[NX - 2][j];
        }
        for (int a = 0; a < NX - 1; ++a) {
            matIn[a][0] = matIn[a][1];
            matIn[a][NY - 1] = matIn[a][NY - 2];
        }
    }
    double __el = MPI_Wtime() - __t0;
    __dacpp_e2e("stencil.ablation", __el);
    return 0;
}

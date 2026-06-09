// decay ablation benchmark source (DSL + e2e timing harness).
// Translate ON:  dacpp.sh translate ... --mpi   (operator-resident)
// Translate OFF: dacpp.sh translate ... --mpi --mpi-no-operator-resident (legacy gather)
// The translator injects MPI_Init/Finalize + the root-rank helper into main();
// we time the whole compute region and print the [MPI TEST] e2e line.
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <mpi.h>
#include "ReconTensor.h"
namespace dacpp {
    typedef std::vector<std::any> list;
}

const double dt = 0.1;
const double T = 100.0;
const size_t numIsotopes = 1024576;

shell dacpp::list DECAY(const dacpp::Vector<double>& N0s,
                        const dacpp::Vector<double>& lambdas,
                        dacpp::Vector<double>& local_A,
                        const dacpp::Vector<double>& t) {
    dacpp::index i;
    dacpp::list dataList{N0s[i], lambdas[i], local_A[i], t[{}]};
    return dataList;
}

calc void decay(double* N0s, double* lambdas, double* local_A, double* t) {
    local_A[0] = N0s[0] * std::exp(-lambdas[0] * t[0]);
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

void calculateDecay(const std::vector<double>& lambdas, const std::vector<double>& N0s,
                    double dt, double T) {
    size_t numIsotopes = lambdas.size();
    std::vector<double> A(static_cast<size_t>(T / dt) * numIsotopes, 0.0);
    std::vector<double> t;
    t.push_back(0.0);
    std::vector<double> local_A(numIsotopes, 0.0);
    dacpp::Vector<double> local_A_tensor(local_A);
    dacpp::Vector<double> N0s_tensor(N0s);
    dacpp::Vector<double> lambdas_tensor(lambdas);
    dacpp::Vector<double> t_tensor(t);
    dacpp::Matrix<double> A_tensor({static_cast<int>(T / dt), static_cast<int>(numIsotopes)}, A);

    while (t_tensor[0] <= T) {
        DECAY(N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor) <-> decay;
        A_tensor[10 * t_tensor[0]] = local_A_tensor;
        t_tensor[0] += dt;
    }
}

int main() {
    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);
    for (size_t i = 0; i < numIsotopes; ++i) lambdas[i] = 0.01 + 0.01 * i;

    double __t0 = MPI_Wtime();
    calculateDecay(lambdas, N0s, dt, T);
    double __el = MPI_Wtime() - __t0;
    __dacpp_e2e("decay.ablation", __el);
    return 0;
}

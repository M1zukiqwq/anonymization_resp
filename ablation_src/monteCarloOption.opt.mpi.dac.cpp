// monteCarloOption perf/correctness source (size matches Tier-1 baseline: 4096^2, 8192 paths/cell).
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <mpi.h>
#include "ReconTensor.h"
using namespace std;
namespace dacpp { typedef std::vector<std::any> list; }

const int NX = 4096;
const int NY = 4096;
const int PATHS_PER_OPTION = 4096;
const int INNER_REPEATS = 2;
const double S0 = 100.0, STRIKE = 100.0, RATE = 0.05, VOLATILITY = 0.2, MATURITY = 1.0;

unsigned int lcgNext(unsigned int state) { return state * 1664525u + 1013904223u; }
double uniform01(unsigned int value) { return ((double)(value & 0x00FFFFFFu) + 1.0) / 16777217.0; }

shell dacpp::list MONTE_CARLO_OPTION(const dacpp::Matrix<int>& seeds,
                                     dacpp::Matrix<double>& prices WRITE) {
    dacpp::index idx1, idx2;
    dacpp::list dataList{seeds[idx1][idx2], prices[idx1][idx2]};
    return dataList;
}

calc void monteCarloOption(int* seed, double* price) {
    unsigned int state = (unsigned int)seed[0] + 1u;
    double payoff_sum = 0.0;
    for (int repeat = 0; repeat < INNER_REPEATS; repeat++) {
        for (int path = 0; path < PATHS_PER_OPTION; path++) {
            state = lcgNext(state); double u1 = uniform01(state);
            state = lcgNext(state); double u2 = uniform01(state);
            double radius = std::sqrt(-2.0 * std::log(u1));
            double angle = 6.2831853071795864769 * u2;
            double normal = radius * std::cos(angle);
            double drift = (RATE - 0.5 * VOLATILITY * VOLATILITY) * MATURITY;
            double diffusion = VOLATILITY * std::sqrt(MATURITY) * normal;
            double terminal = S0 * std::exp(drift + diffusion);
            double payoff = terminal - STRIKE;
            if (payoff < 0.0) payoff = 0.0;
            payoff_sum += payoff;
        }
    }
    price[0] = std::exp(-RATE * MATURITY) * payoff_sum / (PATHS_PER_OPTION * INNER_REPEATS);
}

static void __dacpp_e2e(const char* name, double secs) {
    double mx = 0.0, sm = 0.0;
    MPI_Reduce(&secs, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&secs, &sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    int r = 0, s = 1; MPI_Comm_rank(MPI_COMM_WORLD, &r); MPI_Comm_size(MPI_COMM_WORLD, &s);
    if (r == 0) std::printf("[MPI TEST] %s | ranks=%d | e2e_max=%.6f s | e2e_avg=%.6f s\n", name, s, mx, sm / s);
}

int main() {
    std::vector<int> seeds(NX * NY, 0);
    std::vector<double> prices(NX * NY, 0.0);
    for (int i = 0; i < NY; i++) for (int j = 0; j < NX; j++) seeds[i * NX + j] = i * NX + j;
    dacpp::Matrix<int> seeds_tensor({NY, NX}, seeds);
    dacpp::Matrix<double> prices_tensor({NY, NX}, prices);

    double __t0 = MPI_Wtime();
    MONTE_CARLO_OPTION(seeds_tensor, prices_tensor) <-> monteCarloOption;
    double __el = MPI_Wtime() - __t0;

    int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r);
    if (r == 0) std::printf("[CHECK] price[mid]=%.10f\n", prices_tensor[NY / 2][NX / 2]);
    __dacpp_e2e("monteCarlo.opt", __el);
    return 0;
}

// gradientSum perf/correctness source (8192x8192, matches Tier-1 baseline).
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <mpi.h>
#include "ReconTensor.h"
using namespace std;
namespace dacpp { typedef std::vector<std::any> list; }

#define GRADIENT_NUM_NEURONS 8192
#define GRADIENT_INPUT_SIZE 8192
const int NUM_NEURONS = GRADIENT_NUM_NEURONS;
const int INPUT_SIZE  = GRADIENT_INPUT_SIZE;

shell dacpp::list gradSumShell(dacpp::Matrix<float>& matGrads READ,
                               dacpp::Matrix<float>& matNeuronSum WRITE) {
    dacpp::index idx1, idx2;
    dacpp::list dataList{matGrads[{}][idx1], matNeuronSum[idx1][idx2]};
    return dataList;
}

calc void gradSum(dacpp::Vector<float>& grads, dacpp::Vector<float>& neuronSum) {
    int sum = 0;
    for (int j = 0; j < INPUT_SIZE; ++j) sum += grads[j];
    neuronSum[0] = sum;
}

static void __dacpp_e2e(const char* name, double secs) {
    double mx = 0.0, sm = 0.0;
    MPI_Reduce(&secs, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&secs, &sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    int r = 0, s = 1; MPI_Comm_rank(MPI_COMM_WORLD, &r); MPI_Comm_size(MPI_COMM_WORLD, &s);
    if (r == 0) std::printf("[MPI TEST] %s | ranks=%d | e2e_max=%.6f s | e2e_avg=%.6f s\n", name, s, mx, sm / s);
}

int main() {
    vector<float> host_grads((size_t)NUM_NEURONS * INPUT_SIZE);
    for (int i = 0; i < NUM_NEURONS; i++)
        for (int j = 0; j < INPUT_SIZE; j++) host_grads[(size_t)i * INPUT_SIZE + j] = i + j;
    dacpp::Matrix<float> matGrads({NUM_NEURONS, INPUT_SIZE}, host_grads);
    vector<float> host_neuron_sum(NUM_NEURONS, 0.0f);
    dacpp::Matrix<float> matNeuronSum({NUM_NEURONS, 1}, host_neuron_sum);

    double __t0 = MPI_Wtime();
    gradSumShell(matGrads, matNeuronSum) <-> gradSum;
    double __el = MPI_Wtime() - __t0;

    int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r);
    if (r == 0) std::printf("[CHECK] sum0=%.1f sum1=%.1f\n", (double)matNeuronSum[0][0], (double)matNeuronSum[1][0]);
    __dacpp_e2e("gradientSum.opt", __el);
    return 0;
}

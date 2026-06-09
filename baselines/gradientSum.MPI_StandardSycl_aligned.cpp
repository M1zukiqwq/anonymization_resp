// gradientSum — Tier-1 hand-written MPI+SYCL baseline (aligned).
//
// Aligned with the DACPP-generated RowPartitionFullRow layout:
//   - matGrads is NUM_NEURONS rows x INPUT_SIZE cols (row-major).
//   - The ROWS of matGrads are split across ranks (base/rem block split,
//     identical to the matMul RowPartitionFullRow pattern).
//   - MPI_Scatterv distributes the row-blocks of matGrads
//     (sendcounts[r] = local_rows[r] * INPUT_SIZE).
//   - Each rank's SYCL kernel reduces (sums) each of its local rows over the
//     INPUT_SIZE columns, producing one matNeuronSum entry per row.
//     This mirrors the generated gradSum_mpi_local: an integer accumulator
//     summing grads[j] for j in [0, INPUT_SIZE).
//   - MPI_Gatherv collects the per-row sums back to root (counts = local_rows).
//
// This is a FAIR baseline: same data decomposition + communication pattern as
// the generated code, written cleanly by hand. No DACPP runtime headers.

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
namespace dacpp_sycl = sycl;
#else
#include <CL/sycl.hpp>
namespace dacpp_sycl = cl::sycl;
#endif
#include <mpi.h>
#include "mpi_bench_timer.h"

#include <algorithm>
#include <iostream>
#include <vector>

#ifndef GRADIENT_NUM_NEURONS
#define GRADIENT_NUM_NEURONS 8
#endif

#ifndef GRADIENT_INPUT_SIZE
#define GRADIENT_INPUT_SIZE 8
#endif

static const int NUM_NEURONS = GRADIENT_NUM_NEURONS;
static const int INPUT_SIZE  = GRADIENT_INPUT_SIZE;

// Per-row reduction kernel — mirrors generated gradSum_mpi_local:
//   int sum = 0; for j in [0,INPUT_SIZE): sum += grads[j]; neuronSum[0] = sum;
template <class GradAcc, class SumAcc>
struct GradSumStandardOp {
    GradAcc grads;
    SumAcc  neuron_sum;
    int input_size;

    void operator()(dacpp_sycl::id<1> idx) const {
        const int local_i = static_cast<int>(idx[0]);
        int sum = 0;
        for (int j = 0; j < input_size; ++j) {
            sum += static_cast<int>(grads[local_i * input_size + j]);
        }
        neuron_sum[local_i] = static_cast<float>(sum);
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    BenchTimer t("gradientSum baseline (aligned)");

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // matGrads (NUM_NEURONS x INPUT_SIZE) row-major; rank 0 holds source data.
    // Data init mirrors the generated example main(): host_grads[i][j] = i + j.
    std::vector<float> host_grads;
    if (rank == 0) {
        host_grads.resize(static_cast<std::size_t>(NUM_NEURONS) * INPUT_SIZE);
        for (int i = 0; i < NUM_NEURONS; ++i)
            for (int j = 0; j < INPUT_SIZE; ++j)
                host_grads[i * INPUT_SIZE + j] = static_cast<float>(i + j);
    }

    // Partition matGrads ROWS across ranks (RowPartitionFullRow block split).
    const int base = NUM_NEURONS / size;
    const int rem  = NUM_NEURONS % size;
    const int local_rows = base + (rank < rem ? 1 : 0);

    // --- Scatter row-blocks of matGrads (counts = local_rows * INPUT_SIZE) ---
    std::vector<int> counts_grads(size), displs_grads(size);
    for (int r = 0; r < size; ++r) {
        int lr = base + (r < rem ? 1 : 0);
        int rb = r * base + std::min(r, rem);
        counts_grads[r] = lr * INPUT_SIZE;
        displs_grads[r] = rb * INPUT_SIZE;
    }
    std::vector<float> local_grads(static_cast<std::size_t>(local_rows) * INPUT_SIZE);

    t.comm_begin();
    MPI_Scatterv(rank == 0 ? host_grads.data() : nullptr,
                 counts_grads.data(), displs_grads.data(), MPI_FLOAT,
                 local_grads.data(), local_rows * INPUT_SIZE, MPI_FLOAT,
                 0, MPI_COMM_WORLD);
    t.comm_end();

    // --- Local per-row reduction (one neuronSum entry per local row) ---
    std::vector<float> local_neuron_sum(static_cast<std::size_t>(local_rows), 0.0f);

    if (local_rows > 0) {
        dacpp_sycl::queue q{dacpp_sycl::default_selector_v};

        dacpp_sycl::buffer<float, 1> buf_grads(
            local_grads.data(), dacpp_sycl::range<1>(local_grads.size()));
        dacpp_sycl::buffer<float, 1> buf_sum(
            local_neuron_sum.data(), dacpp_sycl::range<1>(local_neuron_sum.size()));

        t.comp_begin();
        q.submit([&](dacpp_sycl::handler& h) {
            auto grads = buf_grads.get_access<dacpp_sycl::access::mode::read>(h);
            auto neuron_sum = buf_sum.get_access<dacpp_sycl::access::mode::write>(h);

            h.parallel_for<class GradSumStandardKernel>(
                dacpp_sycl::range<1>(local_rows),
                GradSumStandardOp<decltype(grads), decltype(neuron_sum)>{
                    grads, neuron_sum, INPUT_SIZE});
        });
        q.wait();
        t.comp_end();
    }

    // --- Gather per-row sums to rank 0 (counts = local_rows) ---
    std::vector<int> counts_sum(size), displs_sum(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        int lr = base + (r < rem ? 1 : 0);
        counts_sum[r] = lr;
        displs_sum[r] = offset;
        offset += lr;
    }

    std::vector<float> global_neuron_sum;
    if (rank == 0) global_neuron_sum.resize(NUM_NEURONS);

    t.comm_begin();
    MPI_Gatherv(local_neuron_sum.data(), local_rows, MPI_FLOAT,
                rank == 0 ? global_neuron_sum.data() : nullptr,
                counts_sum.data(), displs_sum.data(),
                MPI_FLOAT, 0, MPI_COMM_WORLD);
    t.comm_end();

    if (rank == 0) {
        std::cout << "First 5 neuron gradient sums:\n";
        for (int i = 0; i < std::min(5, NUM_NEURONS); ++i)
            std::cout << global_neuron_sum[i] << " ";
        std::cout << std::endl;
    }

    t.report();

    MPI_Finalize();
    return 0;
}

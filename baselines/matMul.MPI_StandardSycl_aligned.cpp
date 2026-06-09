// matMul Tier-1 hand-written MPI+SYCL baseline (algorithm-aligned).
// Aligned with DACPP generated RowPartitionFullRow:
//   - MPI_Scatterv distributes A's row-blocks across ranks
//   - MPI_Bcast replicates the full B matrix to every rank
//   - local SYCL kernel computes C_rows = A_rows * B
//   - MPI_Gatherv collects C row-blocks to root
// Same decomposition/communication as the generated code; this measures
// code-generation overhead, not an algorithmic difference.
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

#ifndef MATMUL_N
#define MATMUL_N 8192
#endif

template <class AAcc, class BAcc, class CAcc>
struct MatMulOp {
    AAcc a; BAcc b; CAcc c; int local_cols; int inner_dim;
    void operator()(dacpp_sycl::id<2> idx) const {
        const int i = static_cast<int>(idx[0]);
        const int j = static_cast<int>(idx[1]);
        int sum = 0;
        for (int k = 0; k < inner_dim; ++k)
            sum += a[i * inner_dim + k] * b[k * local_cols + j];
        c[i * local_cols + j] = sum;
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    BenchTimer timer("matMul baseline (aligned)");

    constexpr int M = MATMUL_N, K = MATMUL_N, N = MATMUL_N;

    std::vector<int> dataA, dataB(K * N);
    if (rank == 0) {
        dataA.resize(M * K);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) dataA[i * K + k] = (i + k) % 97;
    }
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j) dataB[k * N + j] = (k + j) % 89;

    const int base = M / size, rem = M % size;
    const int local_rows = base + (rank < rem ? 1 : 0);
    const int row_begin  = rank * base + std::min(rank, rem);
    const int local_count = local_rows * N;

    std::vector<int> counts_A(size), displs_A(size), counts_C(size), displs_C(size);
    for (int r = 0; r < size; ++r) {
        int lr = base + (r < rem ? 1 : 0);
        int rb = r * base + std::min(r, rem);
        counts_A[r] = lr * K; displs_A[r] = rb * K;
        counts_C[r] = lr * N; displs_C[r] = rb * N;
    }

    std::vector<int> localA(local_rows * K);
    timer.comm_begin();
    MPI_Scatterv(rank == 0 ? dataA.data() : nullptr, counts_A.data(), displs_A.data(),
                 MPI_INT, localA.data(), local_rows * K, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(dataB.data(), K * N, MPI_INT, 0, MPI_COMM_WORLD);
    timer.comm_end();

    std::vector<int> local_result(local_count, 0);
    if (local_rows > 0) {
        timer.comp_begin();
        dacpp_sycl::queue q{dacpp_sycl::default_selector_v};
        {
            dacpp_sycl::buffer<int, 1> bufA(localA.data(), dacpp_sycl::range<1>(local_rows * K));
            dacpp_sycl::buffer<int, 1> bufB(dataB.data(), dacpp_sycl::range<1>(K * N));
            dacpp_sycl::buffer<int, 1> bufC(local_result.data(), dacpp_sycl::range<1>(local_count));
            q.submit([&](dacpp_sycl::handler& h) {
                auto a = bufA.get_access<dacpp_sycl::access::mode::read>(h);
                auto b = bufB.get_access<dacpp_sycl::access::mode::read>(h);
                auto c = bufC.get_access<dacpp_sycl::access::mode::write>(h);
                h.parallel_for<class MatMulKernel>(
                    dacpp_sycl::range<2>(local_rows, N),
                    MatMulOp<decltype(a), decltype(b), decltype(c)>{a, b, c, N, K});
            });
            q.wait();
        }
        timer.comp_end();
    }

    std::vector<int> global_result;
    if (rank == 0) global_result.resize(M * N);
    timer.comm_begin();
    MPI_Gatherv(local_result.data(), local_count, MPI_INT,
                rank == 0 ? global_result.data() : nullptr,
                counts_C.data(), displs_C.data(), MPI_INT, 0, MPI_COMM_WORLD);
    timer.comm_end();

    if (rank == 0) std::cout << "C[0]=" << global_result[0] << std::endl;
    timer.report();
    MPI_Finalize();
    return 0;
}

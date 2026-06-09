// DFT — Tier-1 hand-written MPI+SYCL baseline (aligned).
//
// Aligned with the DACPP-generated ReplicatedFullTensor layout:
//   - The full input vector is REPLICATED to every rank. The generated code
//     materializes the full input on each rank (no split on the input
//     dimension); here we MPI_Bcast the full input to all ranks.
//   - The OUTPUT frequency bins are split across ranks (base/rem block split):
//     each rank computes a contiguous block of output bins k.
//   - Each output bin k = sum over n in [0,N) of input[n] * W_n, with
//     W_n = (cos(angle), sin(angle)) and angle = -2*pi*k*n/N. This mirrors the
//     generated dft_mpi_local exactly: a single std::complex<double>
//     accumulator (real & imaginary computed together in one pass), driven by
//     the per-bin index k (= "vec[0]" in the generated kernel).
//   - MPI_Gatherv collects the per-rank output blocks back to root.
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
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#ifndef DFT_N
#define DFT_N 8
#endif

static const int N = DFT_N;

// Per-bin DFT kernel — mirrors generated dft_mpi_local:
//   Complex sum(0,0);
//   for n in [0,N): angle = -2*pi*k*n/N; W_n=(cos,sin); sum += input[n]*W_n;
//   output[0] = sum;
template <class InAcc, class OutAcc>
struct DFTStandardOp {
    InAcc  input;
    OutAcc output;
    int    k_begin;
    int    n_size;

    void operator()(dacpp_sycl::id<1> idx) const {
        const int local_k = static_cast<int>(idx[0]);
        const int k = k_begin + local_k;
        double sum_r = 0.0;
        double sum_i = 0.0;
        for (int n = 0; n < n_size; ++n) {
            const double angle =
                -2.0 * 3.1415926535897931 * k * n / static_cast<double>(n_size);
            const double c = dacpp_sycl::cos(angle);
            const double s = dacpp_sycl::sin(angle);
            const double in_r = input[n].real();
            const double in_i = input[n].imag();
            // (in_r + i*in_i) * (c + i*s)
            sum_r += in_r * c - in_i * s;
            sum_i += in_r * s + in_i * c;
        }
        output[local_k] = std::complex<double>(sum_r, sum_i);
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    BenchTimer t("DFT baseline (aligned)");

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Full input vector. Data init mirrors the package/generated baseline:
    // input[n] = (n, 0). Rank 0 fills it; replicated to all via Bcast.
    std::vector<std::complex<double>> input(N);
    if (rank == 0) {
        for (int n = 0; n < N; ++n)
            input[n] = std::complex<double>(static_cast<double>(n), 0.0);
    }

    // --- Replicate full input to all ranks (ReplicatedFullTensor) ---
    t.comm_begin();
    MPI_Bcast(input.data(), N, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    t.comm_end();

    // Split output frequency bins across ranks (block split).
    const int base = N / size;
    const int rem  = N % size;
    const int local_k_count = base + (rank < rem ? 1 : 0);
    const int k_begin = rank * base + std::min(rank, rem);

    std::vector<std::complex<double>> local_output(
        static_cast<std::size_t>(local_k_count));

    if (local_k_count > 0) {
        dacpp_sycl::queue q{dacpp_sycl::default_selector_v};

        dacpp_sycl::buffer<std::complex<double>, 1> buf_in(
            input.data(), dacpp_sycl::range<1>(N));
        dacpp_sycl::buffer<std::complex<double>, 1> buf_out(
            local_output.data(), dacpp_sycl::range<1>(local_k_count));

        t.comp_begin();
        q.submit([&](dacpp_sycl::handler& h) {
            auto in  = buf_in.get_access<dacpp_sycl::access::mode::read>(h);
            auto out = buf_out.get_access<dacpp_sycl::access::mode::write>(h);

            h.parallel_for<class DFTStandardKernel>(
                dacpp_sycl::range<1>(local_k_count),
                DFTStandardOp<decltype(in), decltype(out)>{
                    in, out, k_begin, N});
        });
        q.wait();
        t.comp_end();
    }

    // --- Gather output blocks to rank 0 (counts = local_k_count per rank) ---
    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<std::complex<double>> global_output;
    if (rank == 0) global_output.resize(N);

    t.comm_begin();
    MPI_Gatherv(local_output.data(), local_k_count, MPI_C_DOUBLE_COMPLEX,
                rank == 0 ? global_output.data() : nullptr,
                counts.data(), displs.data(),
                MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    t.comm_end();

    if (rank == 0) {
        std::cout << "DFT[0] = (" << global_output[0].real() << ","
                  << global_output[0].imag() << ")" << std::endl;
    }

    t.report();

    MPI_Finalize();
    return 0;
}

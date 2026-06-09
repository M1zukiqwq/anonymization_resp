// Radioactive decay — MPI + SYCL hand-written Tier-1 baseline (ALIGNED).
//
// This is a FAIR baseline: it uses the EXACT SAME parallel algorithm
// (data decomposition + communication pattern) as the DACPP-generated code
// (example/decay/decay.dac_sycl_buffer.cpp), written cleanly by hand.
//
// Alignment with DACPP generated "Contiguous1D" layout:
//   - The isotope vectors N0s and lambdas are split by index i contiguously
//     across ranks (Contiguous1D). They are SCATTERED ONCE before the time
//     loop (MPI_Scatterv) and stay RESIDENT for every step — exactly as the
//     generated init() scatters them once and run() reuses ctx.local_N0s /
//     ctx.local_lambdas across the while(t<=T) loop.
//   - The scalar time t is replicated to all ranks. The generated run()
//     re-publishes t to every rank on each step, so we MPI_Bcast the scalar t
//     once per step to mirror that per-step communication.
//   - Each step every rank computes local_A[i] = N0s[i]*exp(-lambdas[i]*t)
//     for its owned indices — mirroring decay_mpi_local() in the generated
//     kernel (the value is recomputed from the resident N0s each step, NOT an
//     iterative current*factor recurrence).
//   - local_A is GATHERED to root each step (MPI_Gatherv, root-only output);
//     root stores it as one row of A. Root prints row A[1] for sanity.
//
// Problem sizes and data initialization reuse the package baseline
// (programs/decay1.0/decay_chain.MPI_StandardSycl.cpp): dt, T, numIsotopes,
// N0s=1000, lambdas[i]=0.01+0.01*i.

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
#include <iostream>
#include <vector>

const double dt = 0.1;
const double T = 100.0;
const size_t numIsotopes = 1024576;

// Mirrors decay_mpi_local(): local_A[i] = N0s[i] * exp(-lambdas[i] * t)
template <class N0Acc, class LambdaAcc, class OutAcc>
struct DecayAlignedOp {
    N0Acc n0s;
    LambdaAcc lambdas;
    OutAcc local_a;
    double t;

    void operator()(dacpp_sycl::id<1> idx) const {
        const size_t i = idx[0];
        local_a[i] = n0s[i] * std::exp(-lambdas[i] * t);
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    BenchTimer timer("decay baseline (aligned)");

    const size_t steps = static_cast<size_t>(T / dt);

    // Contiguous1D index split (base/rem) — same as matMul template.
    const size_t base_count = numIsotopes / static_cast<size_t>(size);
    const size_t remainder  = numIsotopes % static_cast<size_t>(size);
    const size_t local_count =
        base_count + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t global_begin =
        static_cast<size_t>(rank) * base_count +
        std::min(static_cast<size_t>(rank), remainder);

    // Per-rank counts/displs for scatter (inputs) and gather (output).
    std::vector<int> counts(static_cast<size_t>(size));
    std::vector<int> displs(static_cast<size_t>(size));
    {
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            const size_t r_count =
                base_count + (static_cast<size_t>(r) < remainder ? 1 : 0);
            counts[static_cast<size_t>(r)] = static_cast<int>(r_count);
            displs[static_cast<size_t>(r)] = offset;
            offset += static_cast<int>(r_count);
        }
    }

    // Root holds the full source N0s / lambdas; they are scattered ONCE.
    std::vector<double> N0s_full;
    std::vector<double> lambdas_full;
    if (rank == 0) {
        N0s_full.assign(numIsotopes, 1000.0);
        lambdas_full.resize(numIsotopes);
        for (size_t i = 0; i < numIsotopes; ++i) {
            lambdas_full[i] = 0.01 + 0.01 * static_cast<double>(i);
        }
    }

    // Resident local inputs (scattered once, reused every step).
    std::vector<double> local_N0s(local_count);
    std::vector<double> local_lambdas(local_count);

    timer.comm_begin();
    MPI_Scatterv(rank == 0 ? N0s_full.data() : nullptr,
                 counts.data(), displs.data(), MPI_DOUBLE,
                 local_N0s.data(), static_cast<int>(local_count), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? lambdas_full.data() : nullptr,
                 counts.data(), displs.data(), MPI_DOUBLE,
                 local_lambdas.data(), static_cast<int>(local_count), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    timer.comm_end();

    std::vector<double> local_A(local_count, 0.0);

    // Root accumulates one gathered row per step.
    std::vector<double> gathered_A;
    std::vector<double> A;
    if (rank == 0) {
        gathered_A.assign(numIsotopes, 0.0);
        A.assign(steps * numIsotopes, 0.0);
    }

    dacpp_sycl::queue q{dacpp_sycl::default_selector_v};

    double t = 0.0;
    while (t <= T) {
        // Replicated scalar t re-published to every rank each step.
        double t_local = t;
        timer.comm_begin();
        MPI_Bcast(&t_local, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        timer.comm_end();

        // Compute local_A[i] = N0s[i]*exp(-lambdas[i]*t) over owned indices.
        if (local_count > 0) {
            timer.comp_begin();
            dacpp_sycl::buffer<double, 1> buf_n0s(local_N0s.data(),
                                                  dacpp_sycl::range<1>(local_count));
            dacpp_sycl::buffer<double, 1> buf_lambdas(local_lambdas.data(),
                                                      dacpp_sycl::range<1>(local_count));
            dacpp_sycl::buffer<double, 1> buf_a(local_A.data(),
                                                dacpp_sycl::range<1>(local_count));
            q.submit([&](dacpp_sycl::handler& h) {
                auto n0s     = buf_n0s.get_access<dacpp_sycl::access::mode::read>(h);
                auto lambdas = buf_lambdas.get_access<dacpp_sycl::access::mode::read>(h);
                auto out     = buf_a.get_access<dacpp_sycl::access::mode::discard_write>(h);
                h.parallel_for<class DecayAlignedKernel>(
                    dacpp_sycl::range<1>(local_count),
                    DecayAlignedOp<decltype(n0s), decltype(lambdas), decltype(out)>{
                        n0s, lambdas, out, t_local});
            });
            q.wait();
            timer.comp_end();
        }

        // Gather local_A to root (root-only output).
        timer.comm_begin();
        MPI_Gatherv(local_A.data(), static_cast<int>(local_count), MPI_DOUBLE,
                    rank == 0 ? gathered_A.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        timer.comm_end();

        const size_t row = static_cast<size_t>(10.0 * t);
        if (rank == 0 && row < steps) {
            for (size_t i = 0; i < numIsotopes; ++i) {
                A[row * numIsotopes + i] = gathered_A[i];
            }
        }

        t += dt;
    }

    // Sanity print: one value from row A[1].
    if (rank == 0 && steps > 1) {
        std::cout << "decay A[1][0] = " << A[1 * numIsotopes + 0] << std::endl;
    }

    timer.report();

    MPI_Finalize();
    return 0;
}

// mdp (Fokker-Planck / drift-diffusion value iteration) — Tier-1 hand-written
// MPI + SYCL baseline (aligned).
//
// Aligned with DACPP generated StencilWindow1D + halo offset=1:
//   - The generated code partitions the 1D grid into contiguous chunks across
//     ranks (one MPI item per interior output cell, RowBlock distribution via
//     get_rank_item_range), holds each chunk plus a 1-cell ghost on each side,
//     and per time step performs a halo route with target offset=1 (publish each
//     rank's freshly-computed boundary values into the neighbor's ghost cell).
//   - The generated local kernel mdp_mpi_local reads a symmetric size-3 window
//     {p[0], p[1], p[2]} = {left, self, right} and writes
//         diffusion = D * (p[2] - 2*p[1] + p[0]) / (dx*dx);
//         drift     = (-A) * (p[2] - p[0]) / (2*dx);
//         new_p[0]  = p[1] + dt * (diffusion + drift);
//     We mirror this update formula EXACTLY.
//   - Sizes / data init / output value (p[2]) taken from the DSL example
//     example/mdp/mdp.dac_sycl_buffer.cpp: N=150, T=1000.
//
// This is a FAIR baseline: same parallel algorithm (1D contiguous decomposition +
// 1-cell neighbor halo exchange per step + SYCL stencil kernel), written cleanly by
// hand. It is NOT a better algorithm and NOT a strawman; the only intended delta
// versus the generated code is code-generation overhead.

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

// Constants mirror the DSL example exactly.
static const double A  = 1.0;  // drift / attraction coefficient
static const double D  = 0.1;  // diffusion coefficient
static const double dx = 0.1;  // spatial step
static const double dt = 0.01; // time step
static const int    N  = 1000000;
static const int    T  = 20000;

// SYCL functor mirroring the generated mdp_mpi_local kernel.
// View is {left, self, right}: acc_p[li-1]=p[0], acc_p[li]=p[1], acc_p[li+1]=p[2].
template <class InAcc, class OutAcc>
struct MDPAlignedOp {
    InAcc  acc_p;
    OutAcc acc_np;

    void operator()(dacpp_sycl::id<1> idx) const {
        const int li = static_cast<int>(idx[0]) + 1; // skip left halo
        const double diffusion =
            D * (acc_p[li + 1] - 2.0 * acc_p[li] + acc_p[li - 1]) / (dx * dx);
        const double drift = (-A) * (acc_p[li + 1] - acc_p[li - 1]) / (2.0 * dx);
        acc_np[li] = acc_p[li] + dt * (diffusion + drift);
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    BenchTimer t("mdp baseline (aligned)");

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Generated code updates interior cells 1..N-2 (N-2 output items),
    // one MPI item per cell, contiguous RowBlock partition across ranks.
    const int interior = N - 2;
    if (size > interior) {
        if (rank == 0) std::cerr << "MPI size too large\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int base        = interior / size;
    const int rem         = interior % size;
    const int local_count = base + (rank < rem ? 1 : 0);
    const int local_begin = 1 + rank * base + std::min(rank, rem); // global index of first owned cell

    const int prev = rank > 0        ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;

    // Local arrays: 1 left ghost + local_count owned + 1 right ghost (halo width 1).
    const int local_ext = local_count + 2;
    std::vector<double> p(local_ext, 0.0);
    std::vector<double> new_p(local_ext, 0.0);

    // Initialize from the global initializer (computed locally per cell):
    //   p[i] = exp(-(i*dx - 5)^2 / 2)
    auto init_gauss = [](int gi) -> double {
        double x = gi * dx;
        return std::exp(-std::pow(x - 5.0, 2) / 2.0);
    };
    for (int li = 0; li < local_count; ++li) {
        p[li + 1] = init_gauss(local_begin + li);
    }
    // Seed boundary ghosts on the outermost ranks from the global grid edges.
    if (rank == 0)        p[0]               = init_gauss(local_begin - 1);
    if (rank == size - 1) p[local_count + 1] = init_gauss(local_begin + local_count);

    dacpp_sycl::queue q_dev{dacpp_sycl::default_selector_v};

    for (int t_step = 0; t_step < T; ++t_step) {
        // (a) COMMUNICATION: exchange 1-cell halo with left/right neighbors.
        t.comm_begin();
        double send_left  = p[1];
        double send_right = p[local_count];
        double recv_left  = 0.0;
        double recv_right = 0.0;
        MPI_Sendrecv(&send_left,  1, MPI_DOUBLE, prev, 0,
                     &recv_right, 1, MPI_DOUBLE, next, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, next, 1,
                     &recv_left,  1, MPI_DOUBLE, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (prev != MPI_PROC_NULL) p[0]               = recv_left;
        if (next != MPI_PROC_NULL) p[local_count + 1] = recv_right;
        t.comm_end();

        // (b) COMPUTATION: SYCL stencil kernel over the owned cells.
        t.comp_begin();
        {
            dacpp_sycl::buffer<double, 1> buf_p(p.data(), dacpp_sycl::range<1>(local_ext));
            dacpp_sycl::buffer<double, 1> buf_np(new_p.data(), dacpp_sycl::range<1>(local_ext));
            q_dev.submit([&](dacpp_sycl::handler& h) {
                auto acc_p  = buf_p.get_access<dacpp_sycl::access::mode::read>(h);
                auto acc_np = buf_np.get_access<dacpp_sycl::access::mode::write>(h);
                h.parallel_for<class MDPAlignedKernel>(
                    dacpp_sycl::range<1>(local_count),
                    MDPAlignedOp<decltype(acc_p), decltype(acc_np)>{acc_p, acc_np});
            });
            q_dev.wait();
        }
        t.comp_end();

        // (c) swap buffers for the next step.
        std::swap(p, new_p);
    }

    // Gather owned cells to root for a sanity value (p[2], matching the DSL example).
    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
    std::vector<double> local_out(p.begin() + 1, p.begin() + 1 + local_count);
    std::vector<double> global_p;
    if (rank == 0) global_p.resize(N, 0.0);
    MPI_Gatherv(local_out.data(), local_count, MPI_DOUBLE,
                rank == 0 ? global_p.data() + 1 : nullptr,
                counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << global_p[2] << std::endl;
    }

    t.report();
    MPI_Finalize();
    return 0;
}

// liuliang (LWR traffic-flow) — Tier-1 hand-written MPI + SYCL baseline (aligned).
//
// Aligned with DACPP generated StencilWindow1D + halo offset=1:
//   - The generated code partitions the 1D density grid into contiguous chunks
//     across ranks (one MPI item per interior output cell, RowBlock distribution
//     via get_rank_item_range), holds each chunk plus a 1-cell ghost on each side,
//     and per time step performs a halo route with target offset=1 (publish each
//     rank's freshly-computed boundary values into the neighbor's ghost cell).
//   - The generated local kernel lwr_mpi_local reads a size-2 window
//     {rho[0], rho[1]} = {left-neighbor, self} and writes
//         new_rho[0] = rho[1] - (DELTA_T/DELTA_X) * (q(rho[1]) - q(rho[0]));
//         new_rho[0] = max(0, new_rho[0]);
//     We mirror this update formula EXACTLY (left-biased upwind LWR flux).
//   - Sizes / data init / output value (rho[15]) taken from the DSL example
//     example/liuliang/liuliang.dac_sycl_buffer.cpp: WIDTH=100, TIME_STEPS=200.
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
#include <cmath>
#include "mpi_bench_timer.h"

#include <algorithm>
#include <iostream>
#include <vector>

// Sizes mirror the DSL example exactly.
static const int    WIDTH      = 1000000;
static const int    TIME_STEPS = 5000;
static const double DELTA_T    = 0.01;
static const double DELTA_X    = 1.0;

// Traffic-flow flux function q(rho), identical to the DSL example.
static inline double q(double rho) {
    const double V_max   = 30.0;
    const double rho_max = 50.0;
    return rho * V_max * (1.0 - rho / rho_max);
}

// SYCL functor mirroring the generated lwr_mpi_local kernel.
// View is {left-neighbor, self}: acc_rho[li-1] = rho[0], acc_rho[li] = rho[1].
template <class InAcc, class OutAcc>
struct LiuliangAlignedOp {
    InAcc  acc_rho;
    OutAcc acc_new;

    void operator()(dacpp_sycl::id<1> idx) const {
        const int li = static_cast<int>(idx[0]) + 1; // skip left halo
        const double flow_self = q(acc_rho[li]);     // q(rho[1])
        const double flow_left = q(acc_rho[li - 1]); // q(rho[0])
        double v = acc_rho[li] - (DELTA_T / DELTA_X) * (flow_self - flow_left);
        acc_new[li] = std::max(0.0, v);
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    BenchTimer t("liuliang baseline (aligned)");

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Generated code updates interior cells 1..WIDTH-2 (WIDTH-2 output items),
    // one MPI item per cell, contiguous RowBlock partition across ranks.
    const int interior = WIDTH - 2;
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
    std::vector<double> rho(local_ext, 0.0);
    std::vector<double> new_rho(local_ext, 0.0);

    // Initialize density from the global initializer (computed locally per cell).
    auto init_density = [](int gi) -> double {
        if (gi < WIDTH / 4)         return 40.0;
        else if (gi < 3 * WIDTH / 4) return 20.0;
        else                        return 10.0;
    };
    for (int li = 0; li < local_count; ++li) {
        rho[li + 1] = init_density(local_begin + li);
    }
    // Seed boundary ghosts on the outermost ranks from the global grid edges.
    if (rank == 0)        rho[0]               = init_density(local_begin - 1);
    if (rank == size - 1) rho[local_count + 1] = init_density(local_begin + local_count);

    dacpp_sycl::queue q_dev{dacpp_sycl::default_selector_v};

    for (int t_step = 0; t_step < TIME_STEPS; ++t_step) {
        // (a) COMMUNICATION: exchange 1-cell halo with left/right neighbors.
        t.comm_begin();
        double send_left  = rho[1];
        double send_right = rho[local_count];
        double recv_left  = 0.0;
        double recv_right = 0.0;
        MPI_Sendrecv(&send_left,  1, MPI_DOUBLE, prev, 0,
                     &recv_right, 1, MPI_DOUBLE, next, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, next, 1,
                     &recv_left,  1, MPI_DOUBLE, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (prev != MPI_PROC_NULL) rho[0]               = recv_left;
        if (next != MPI_PROC_NULL) rho[local_count + 1] = recv_right;
        t.comm_end();

        // (b) COMPUTATION: SYCL stencil kernel over the owned cells.
        t.comp_begin();
        {
            dacpp_sycl::buffer<double, 1> buf_rho(rho.data(), dacpp_sycl::range<1>(local_ext));
            dacpp_sycl::buffer<double, 1> buf_new(new_rho.data(), dacpp_sycl::range<1>(local_ext));
            q_dev.submit([&](dacpp_sycl::handler& h) {
                auto acc_rho = buf_rho.get_access<dacpp_sycl::access::mode::read>(h);
                auto acc_new = buf_new.get_access<dacpp_sycl::access::mode::write>(h);
                h.parallel_for<class LiuliangAlignedKernel>(
                    dacpp_sycl::range<1>(local_count),
                    LiuliangAlignedOp<decltype(acc_rho), decltype(acc_new)>{acc_rho, acc_new});
            });
            q_dev.wait();
        }
        t.comp_end();

        // (c) swap buffers: new_rho becomes rho for the next step (interior cells).
        for (int li = 1; li <= local_count; ++li) rho[li] = new_rho[li];
    }

    // Gather owned cells to root for a sanity value (rho[15], matching the DSL example).
    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
    std::vector<double> local_out(rho.begin() + 1, rho.begin() + 1 + local_count);
    std::vector<double> global_rho;
    if (rank == 0) global_rho.resize(WIDTH, 0.0);
    MPI_Gatherv(local_out.data(), local_count, MPI_DOUBLE,
                rank == 0 ? global_rho.data() + 1 : nullptr,
                counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << global_rho[15] << std::endl;
    }

    t.report();
    MPI_Finalize();
    return 0;
}

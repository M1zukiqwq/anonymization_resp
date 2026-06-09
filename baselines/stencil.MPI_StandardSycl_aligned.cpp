// stencil — Tier-1 hand-written MPI+SYCL baseline (aligned).
//
// Aligned with DACPP generated StencilWindow2D + resident-halo, halo offset=(1,1):
//   - Grid is NX x NY (row-major). The NX ROWS are split into contiguous
//     row-blocks across ranks (base/rem block split). Each rank owns
//     local_rows rows PLUS one ghost ROW on top and one on bottom (halo
//     width 1; offset=(1,1) => exactly 1 ghost row per side, NY elems each).
//   - Each time step:
//       (a) COMMUNICATION: exchange the top/bottom ghost ROW (NY doubles)
//           with the prev/next neighbor rank via MPI_Sendrecv. This mirrors
//           the generated halo-exchange (resident-halo: the row-partitioned
//           tensor stays distributed across steps; only the 1-row halo moves).
//       (b) COMPUTATION: a SYCL kernel applies the 5-point heat (Laplacian)
//           update over the rank's owned interior cells, reading the ghost
//           rows for the up/down neighbors. The update formula is copied
//           EXACTLY from the generated stencil_mpi_local:
//               out = c + alpha*delta_t*((down - 2c + up)/(dx*dx)
//                                       +(right - 2c + left)/(dy*dy))
//           where c=mat[1][1], up=mat[0][1], down=mat[2][1],
//           left=mat[1][0], right=mat[1][2].
//       (c) Domain-boundary cells (global row 0 / NX-1, col 0 / NY-1) are
//           copied from their inward neighbor (mirrors the generated
//           boundary-local slot plans), then swap in/out.
//   - After the loop, MPI_Gatherv collects all owned rows to root and prints
//     row 0 (matIn[0]) — the same sanity value the generated example prints.
//
// FAIR baseline: identical decomposition + halo-exchange communication pattern
// + update formula as the generated code, written cleanly by hand. No DACPP
// runtime headers.

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
#include <cstddef>
#include <iostream>
#include <vector>

#ifndef STENCIL_NX
#define STENCIL_NX 8
#endif

#ifndef STENCIL_NY
#define STENCIL_NY 8
#endif

#ifndef STENCIL_TIME_STEPS
#define STENCIL_TIME_STEPS 10
#endif

static const int NX = STENCIL_NX;
static const int NY = STENCIL_NY;
static const int TIME_STEPS = STENCIL_TIME_STEPS;
static const double Lx = 10.0;
static const double Ly = 10.0;
static const double alpha = 0.01;

static int rows_for_rank(int rank, int size) {
    const int base = NX / size;
    const int rem = NX % size;
    return base + (rank < rem ? 1 : 0);
}

static int row_begin_for_rank(int rank, int size) {
    const int base = NX / size;
    const int rem = NX % size;
    return rank * base + std::min(rank, rem);
}

// Interior 5-point heat update — mirrors generated stencil_mpi_local exactly.
// Operates on the [local_rows] x [NY-2] interior of each rank's owned block;
// rows are laid out with 1 ghost row above (index 0) and below.
template <class CurrAcc, class NextAcc>
struct StencilHeatOp {
    CurrAcc curr;
    NextAcc next_acc;
    int row_begin;
    int nx;
    int ny;
    std::size_t pitch;
    double dx;
    double dy;
    double delta_t;
    double alpha_coeff;

    void operator()(dacpp_sycl::id<2> idx) const {
        const int lr = static_cast<int>(idx[0]) + 1; // local row incl. ghost offset
        const int j = static_cast<int>(idx[1]) + 1;  // interior col [1, NY-2]
        const int gi = row_begin + lr - 1;           // global row index
        // Global interior only; domain boundary rows handled separately.
        if (gi <= 0 || gi >= nx - 1) {
            return;
        }
        const std::size_t center = static_cast<std::size_t>(lr) * pitch + j;
        // mat[1][1]=center, mat[2][1]=down, mat[0][1]=up, mat[1][2]=right, mat[1][0]=left
        const double u_xx =
            (curr[center + pitch] - 2.0 * curr[center] + curr[center - pitch]) /
            (dx * dx);
        const double u_yy =
            (curr[center + 1] - 2.0 * curr[center] + curr[center - 1]) /
            (dy * dy);
        next_acc[center] =
            curr[center] + alpha_coeff * delta_t * (u_xx + u_yy);
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    BenchTimer t("stencil baseline (aligned)");

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > NX) {
        if (rank == 0) std::cerr << "MPI size must be <= STENCIL_NX\n";
        t.report();
        MPI_Finalize();
        return 1;
    }

    const int local_rows = rows_for_rank(rank, size);
    const int row_begin = row_begin_for_rank(rank, size);
    const int prev = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;

    const double dx = Lx / (NX - 1);
    const double dy = Ly / (NY - 1);
    const double dt_stability =
        (dx * dx * dy * dy) / (2.0 * alpha * (dx * dx + dy * dy));
    const double delta_t = 0.4 * dt_stability;

    const std::size_t pitch = static_cast<std::size_t>(NY);
    const std::size_t local_extent =
        static_cast<std::size_t>(local_rows + 2) * pitch; // +2 ghost rows
    const int nx = NX;
    const int ny = NY;
    const double alpha_coeff = alpha;

    std::vector<double> local_curr(local_extent, 0.0);
    std::vector<double> local_next(local_extent, 0.0);

    // Initial condition — identical Gaussian to the generated example main():
    //   u_curr[i][j] = exp(-((x-Lx/2)^2 + (y-Ly/2)^2)/(2*sigma^2)), sigma=1.
    const double sigma = 1.0;
    for (int lr = 1; lr <= local_rows; ++lr) {
        const int gi = row_begin + lr - 1;
        for (int j = 0; j < NY; ++j) {
            const double x = gi * dx;
            const double y = j * dy;
            local_curr[static_cast<std::size_t>(lr) * pitch + j] =
                std::exp(-((x - Lx / 2.0) * (x - Lx / 2.0) +
                           (y - Ly / 2.0) * (y - Ly / 2.0)) /
                         (2.0 * sigma * sigma));
        }
    }

    dacpp_sycl::queue q{dacpp_sycl::default_selector_v};

    for (int step = 0; step < TIME_STEPS; ++step) {
        // (a) COMMUNICATION: halo-row exchange with top/bottom neighbors.
        t.comm_begin();
        // Send my top owned row up; receive into my top ghost row.
        MPI_Sendrecv(local_curr.data() + pitch, NY, MPI_DOUBLE, prev, 10,
                     local_curr.data(), NY, MPI_DOUBLE, prev, 11,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Send my bottom owned row down; receive into my bottom ghost row.
        MPI_Sendrecv(local_curr.data() + static_cast<std::size_t>(local_rows) * pitch,
                     NY, MPI_DOUBLE, next, 11,
                     local_curr.data() + static_cast<std::size_t>(local_rows + 1) * pitch,
                     NY, MPI_DOUBLE, next, 10,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        t.comm_end();

        // (b) COMPUTATION: 5-point heat update over owned interior cells.
        t.comp_begin();
        {
            dacpp_sycl::buffer<double, 1> curr_buf(
                local_curr.data(), dacpp_sycl::range<1>(local_curr.size()));
            dacpp_sycl::buffer<double, 1> next_buf(
                local_next.data(), dacpp_sycl::range<1>(local_next.size()));
            q.submit([&](dacpp_sycl::handler& h) {
                auto curr = curr_buf.get_access<dacpp_sycl::access::mode::read>(h);
                auto next_acc = next_buf.get_access<dacpp_sycl::access::mode::read_write>(h);
                h.parallel_for<class StencilHeatKernel>(
                    dacpp_sycl::range<2>(static_cast<std::size_t>(local_rows),
                                         static_cast<std::size_t>(NY - 2)),
                    StencilHeatOp<decltype(curr), decltype(next_acc)>{
                        curr, next_acc, row_begin, nx, ny, pitch, dx, dy,
                        delta_t, alpha_coeff});
            });
            q.wait();
        }
        t.comp_end();

        // (c) Domain-boundary copy (mirrors generated boundary-local slot
        // plans: row 0<-row 1, row NX-1<-row NX-2, col 0<-col 1, col NY-1<-col NY-2).
        for (int lr = 1; lr <= local_rows; ++lr) {
            const int gi = row_begin + lr - 1;
            const std::size_t row = static_cast<std::size_t>(lr) * pitch;
            // Left/right column boundaries on every owned row.
            local_next[row] = local_next[row + 1];
            local_next[row + ny - 1] = local_next[row + ny - 2];
            // Top global boundary row 0 copies from global row 1.
            if (gi == 0) {
                for (int j = 0; j < NY; ++j)
                    local_next[row + j] = local_next[row + pitch + j];
            }
            // Bottom global boundary row NX-1 copies from global row NX-2.
            if (gi == nx - 1) {
                for (int j = 0; j < NY; ++j)
                    local_next[row + j] = local_next[row - pitch + j];
            }
        }

        std::swap(local_curr, local_next);
    }

    // Gather all owned rows to root.
    std::vector<int> counts, displs;
    std::vector<double> global_grid;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = rows_for_rank(r, size) * NY;
            displs[r] = row_begin_for_rank(r, size) * NY;
        }
        global_grid.resize(static_cast<std::size_t>(NX) * NY);
    }

    t.comm_begin();
    MPI_Gatherv(local_curr.data() + pitch, local_rows * NY, MPI_DOUBLE,
                rank == 0 ? global_grid.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    t.comm_end();

    // Sanity output: row 0 of the final grid (mirrors generated matIn[0].print()).
    if (rank == 0) {
        double row0_checksum = 0.0;
        std::cout << "row0:";
        for (int j = 0; j < NY; ++j) {
            std::cout << " " << global_grid[j];
            row0_checksum += global_grid[j];
        }
        std::cout << "\nrow0_checksum=" << row0_checksum << std::endl;
    }

    t.report();

    MPI_Finalize();
    return 0;
}

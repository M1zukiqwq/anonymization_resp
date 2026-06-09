// waveEquation — Tier-1 hand-written MPI+SYCL baseline (aligned).
//
// Aligned with DACPP generated StencilWindow2D + resident-halo + direct-reader
// (buffer-swap), halo offset=(1,1):
//   - Grid is NX x NY (row-major). The NX ROWS are split into contiguous
//     row-blocks across ranks (base/rem block split). Each rank owns
//     local_rows rows PLUS one ghost ROW on top and bottom (halo width 1;
//     offset=(1,1) => exactly 1 ghost row per side, NY elems each).
//   - THREE buffers prev / curr / next are kept resident and rotated each step
//     (mirrors the generated direct-reader buffer-swap: next<-computed,
//     curr->prev, next->curr via the read-cache state transition).
//   - Each time step:
//       (a) COMMUNICATION: exchange the top/bottom ghost ROW of CURR (NY
//           doubles) with prev/next neighbor ranks via MPI_Sendrecv. Only the
//           1-row halo of the resident row partition moves (resident-halo).
//       (b) COMPUTATION: a SYCL kernel applies the wave update over the rank's
//           owned interior cells, reading ghost rows for up/down neighbors.
//           Formula copied EXACTLY from generated waveEq_mpi_local:
//               u_xx = (down - 2c + up)/(dx*dx)
//               u_yy = (right - 2c + left)/(dy*dy)
//               next = 2c - prev + (c*c)*dt*dt*(u_xx + u_yy)
//           with c=cur[1][1], up=cur[0][1], down=cur[2][1],
//           left=cur[1][0], right=cur[1][2]; dt = 0.5*min(dx,dy)/c.
//           Global domain boundary cells stay 0 (Dirichlet), matching the
//           generated boundary-local plans on cur (rows 0/NX-1, cols 0/NY-1).
//       (c) ROTATE: prev<-curr, curr<-next (buffer swap).
//   - After the loop, MPI_Gatherv collects all owned rows of curr to root and
//     prints global_out[0] — the same sanity value the package baseline prints.
//
// FAIR baseline: identical decomposition + halo-row exchange + 3-buffer
// rotation + update formula as the generated code, written cleanly by hand.
// No DACPP runtime headers.

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

#ifndef WAVE_NX
#define WAVE_NX 8
#endif

#ifndef WAVE_NY
#define WAVE_NY 8
#endif

#ifndef WAVE_TIME_STEPS
#define WAVE_TIME_STEPS 10
#endif

static const int NX = WAVE_NX;
static const int NY = WAVE_NY;
static const int TIME_STEPS = WAVE_TIME_STEPS;
static const double Lx = 10.0;
static const double Ly = 10.0;
static const double c = 1.0;

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

// Wave update — mirrors generated waveEq_mpi_local exactly.
// Each rank's owned block has 1 ghost row above (index 0) and below.
template <class PrevAcc, class CurrAcc, class NextAcc>
struct WaveEqOp {
    PrevAcc d_prev;
    CurrAcc d_curr;
    NextAcc d_next;
    int global_begin;
    int nx;
    int ny;
    std::size_t pitch;
    double dx;
    double dy;
    double dt;
    double wave_c;

    void operator()(dacpp_sycl::id<2> idx) const {
        const int lr = static_cast<int>(idx[0]) + 1; // local row incl. ghost offset
        const int j = static_cast<int>(idx[1]);       // col [0, NY-1]
        const int gi = global_begin + lr - 1;          // global row index
        const std::size_t pos = static_cast<std::size_t>(lr) * pitch + j;

        // Global domain boundary stays 0 (Dirichlet), matching the generated
        // boundary-local plans (rows 0/NX-1, cols 0/NY-1).
        if (gi == 0 || gi == nx - 1 || j == 0 || j == ny - 1) {
            d_next[pos] = 0.0;
            return;
        }

        const double center = d_curr[pos];
        const double u_xx =
            (d_curr[pos + pitch] - 2.0 * center + d_curr[pos - pitch]) /
            (dx * dx);
        const double u_yy =
            (d_curr[pos + 1] - 2.0 * center + d_curr[pos - 1]) /
            (dy * dy);
        d_next[pos] =
            2.0 * center - d_prev[pos] +
            wave_c * wave_c * dt * dt * (u_xx + u_yy);
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    BenchTimer t("waveEquation baseline (aligned)");

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > NX) {
        if (rank == 0) std::cerr << "MPI size must be <= WAVE_NX\n";
        t.report();
        MPI_Finalize();
        return 1;
    }

    const int local_rows = rows_for_rank(rank, size);
    const int global_begin = row_begin_for_rank(rank, size);
    const int up = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int down = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;

    const std::size_t pitch = static_cast<std::size_t>(NY);
    const std::size_t local_with_halo =
        static_cast<std::size_t>(local_rows + 2) * pitch; // +2 ghost rows

    const double dx = Lx / (NX - 1);
    const double dy = Ly / (NY - 1);
    const double dt = 0.5 * std::fmin(dx, dy) / c;
    const int nx = NX;
    const int ny = NY;
    const double wave_c = c;

    // Three resident buffers (prev/curr/next) rotated each step.
    std::vector<double> local_prev(local_with_halo, 0.0);
    std::vector<double> local_curr(local_with_halo, 0.0);
    std::vector<double> local_next(local_with_halo, 0.0);

    // Initial condition — identical Gaussian to the package baseline:
    //   prev[i][j] = exp(-((x-Lx/2)^2 + (y-Ly/2)^2)/(2*sigma^2)), sigma=0.5.
    const double sigma = 0.5;
    for (int lr = 0; lr < local_rows; ++lr) {
        const int gi = global_begin + lr;
        for (int j = 0; j < NY; ++j) {
            const double x = gi * dx;
            const double y = j * dy;
            local_prev[static_cast<std::size_t>(lr + 1) * pitch + j] =
                std::exp(-((x - Lx / 2.0) * (x - Lx / 2.0) +
                           (y - Ly / 2.0) * (y - Ly / 2.0)) /
                         (2.0 * sigma * sigma));
        }
    }

    dacpp_sycl::queue q{dacpp_sycl::default_selector_v};

    for (int step = 0; step < TIME_STEPS; ++step) {
        // (a) COMMUNICATION: halo-row exchange of CURR with top/bottom neighbors.
        t.comm_begin();
        // Send my top owned row up; receive into my top ghost row.
        MPI_Sendrecv(local_curr.data() + pitch, NY, MPI_DOUBLE, up, 0,
                     local_curr.data(), NY, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Send my bottom owned row down; receive into my bottom ghost row.
        MPI_Sendrecv(local_curr.data() + static_cast<std::size_t>(local_rows) * pitch,
                     NY, MPI_DOUBLE, down, 1,
                     local_curr.data() + static_cast<std::size_t>(local_rows + 1) * pitch,
                     NY, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        t.comm_end();

        // (b) COMPUTATION: wave update over owned cells.
        t.comp_begin();
        {
            dacpp_sycl::buffer<double, 1> prev_buf(
                local_prev.data(), dacpp_sycl::range<1>(local_prev.size()));
            dacpp_sycl::buffer<double, 1> curr_buf(
                local_curr.data(), dacpp_sycl::range<1>(local_curr.size()));
            dacpp_sycl::buffer<double, 1> next_buf(
                local_next.data(), dacpp_sycl::range<1>(local_next.size()));
            q.submit([&](dacpp_sycl::handler& h) {
                auto prev_acc = prev_buf.get_access<dacpp_sycl::access::mode::read>(h);
                auto curr_acc = curr_buf.get_access<dacpp_sycl::access::mode::read>(h);
                auto next_acc = next_buf.get_access<dacpp_sycl::access::mode::write>(h);
                h.parallel_for<class WaveEqKernel>(
                    dacpp_sycl::range<2>(static_cast<std::size_t>(local_rows),
                                         static_cast<std::size_t>(NY)),
                    WaveEqOp<decltype(prev_acc), decltype(curr_acc),
                             decltype(next_acc)>{
                        prev_acc, curr_acc, next_acc, global_begin, nx, ny,
                        pitch, dx, dy, dt, wave_c});
            });
            q.wait();
        }
        t.comp_end();

        // (c) ROTATE buffers: prev<-curr, curr<-next (buffer-swap).
        std::swap(local_prev, local_curr);
        std::swap(local_curr, local_next);
    }

    // Gather owned rows of curr to root.
    std::vector<double> local_out(static_cast<std::size_t>(local_rows) * pitch);
    std::copy(local_curr.begin() + static_cast<std::ptrdiff_t>(pitch),
              local_curr.begin() + static_cast<std::ptrdiff_t>(pitch + local_out.size()),
              local_out.begin());

    std::vector<int> counts, displs;
    std::vector<double> global_out;
    if (rank == 0) {
        counts.resize(size, 0);
        displs.resize(size, 0);
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            counts[r] = rows_for_rank(r, size) * NY;
            displs[r] = offset;
            offset += counts[r];
        }
        global_out.resize(static_cast<std::size_t>(NX) * NY);
    }

    t.comm_begin();
    MPI_Gatherv(local_out.data(), static_cast<int>(local_out.size()), MPI_DOUBLE,
                rank == 0 ? global_out.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    t.comm_end();

    // Sanity output: global_out[0] (mirrors the package baseline print).
    if (rank == 0) {
        double row0_checksum = 0.0;
        for (int j = 0; j < NY; ++j) row0_checksum += global_out[j];
        std::cout << global_out[0] << std::endl;
        std::cout << "row0_checksum=" << row0_checksum << std::endl;
    }

    t.report();

    MPI_Finalize();
    return 0;
}

// Monte Carlo Option Pricing — MPI + SYCL standard implementation
// Tier-1 hand-written baseline, ALGORITHM-ALIGNED with the DACPP generated code.
//
// Aligned with DACPP generated RowBlock2D:
//   - The problem is a 2D grid of NY x NX option cells (one seed per cell).
//   - DACPP splits the work over the two bind sets (dim0=NY rows, dim1=NX cols),
//     so the flattened item space is NY*NX cells, partitioned into contiguous
//     blocks across ranks (get_rank_item_range -> base/rem block split). Since
//     the items are laid out row-major (i*NX+j), this is a contiguous row-block
//     2D partition: each rank owns a contiguous chunk of cells.
//   - seeds (int) are SCATTERED to the owning ranks (MPI_Scatterv, units = cells).
//   - each cell runs the Monte Carlo simulation INDEPENDENTLY (no halo, no Bcast
//     of shared data) — DACPP issues no MPI_Bcast for this kernel.
//   - prices (double) are GATHERED to root (MPI_Gatherv, units = cells).
//   - root prints prices[NY/2][NX/2].
// Counts/displacements are in units of cells (each cell = 1 seed in / 1 price out),
// matching the generated pack/scatter/gather over the flattened cell index space.
//
// The per-cell math and the LCG RNG are copied verbatim from the generated
// monteCarloOption_mpi_local kernel so the numerics are bit-identical.

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
namespace dacpp_sycl = sycl;
#else
#include <CL/sycl.hpp>
namespace dacpp_sycl = cl::sycl;
#endif
#include <mpi.h>
#include <algorithm>
#include "mpi_bench_timer.h"

#include <vector>
#include <cmath>
#include <iostream>

// ---- Problem sizes / parameters (from generated example main()) ----
static const int NX = 4096;
static const int NY = 4096;
static const int PATHS_PER_OPTION = 4096;
static const int INNER_REPEATS = 2;

static const double S0 = 100.0;
static const double STRIKE = 100.0;
static const double RATE = 0.05;
static const double VOLATILITY = 0.2;
static const double MATURITY = 1.0;

// ---- Inline LCG RNG, identical to generated kernel ----
inline unsigned int lcgNext(unsigned int state) {
    return state * 1664525u + 1013904223u;
}
inline double uniform01(unsigned int value) {
    return ((double)(value & 0x00FFFFFFu) + 1.0) / 16777217.0;
}

// ---- SYCL functor: one work-item per local cell, Monte Carlo pricing ----
template <class SeedAcc, class PriceAcc>
struct MonteCarloOptionOp {
    SeedAcc seed;
    PriceAcc price;

    void operator()(dacpp_sycl::id<1> idx) const {
        const int cell = static_cast<int>(idx[0]);
        unsigned int state = (unsigned int)seed[cell] + 1U;
        double payoff_sum = 0.0;
        for (int repeat = 0; repeat < INNER_REPEATS; ++repeat) {
            for (int path = 0; path < PATHS_PER_OPTION; ++path) {
                state = lcgNext(state);
                double u1 = uniform01(state);
                state = lcgNext(state);
                double u2 = uniform01(state);
                double radius = std::sqrt(-2.0 * std::log(u1));
                double angle = 6.2831853071795862 * u2;
                double normal = radius * std::cos(angle);
                double drift = (RATE - 0.5 * VOLATILITY * VOLATILITY) * MATURITY;
                double diffusion = VOLATILITY * std::sqrt(MATURITY) * normal;
                double terminal = S0 * std::exp(drift + diffusion);
                double payoff = terminal - STRIKE;
                if (payoff < 0.0) {
                    payoff = 0.0;
                }
                payoff_sum += payoff;
            }
        }
        price[cell] = std::exp(-RATE * MATURITY) * payoff_sum /
                      (PATHS_PER_OPTION * INNER_REPEATS);
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    BenchTimer t("monteCarloOption baseline (aligned)");

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int total_cells = NY * NX;

    // Global seeds on root: seeds[i*NX+j] = i*NX+j  (from generated main()).
    std::vector<int> seeds;
    if (rank == 0) {
        seeds.resize(total_cells);
        for (int i = 0; i < NY; ++i)
            for (int j = 0; j < NX; ++j)
                seeds[i * NX + j] = i * NX + j;
    }

    // --- RowBlock2D contiguous block split of the flattened cell space ---
    const int base = total_cells / size;
    const int rem  = total_cells % size;
    const int local_cells = base + (rank < rem ? 1 : 0);

    // counts/displs in units of cells (1 seed in / 1 price out per cell),
    // matching the generated per-item scatter/gather.
    std::vector<int> counts(size), displs(size);
    {
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            counts[r] = base + (r < rem ? 1 : 0);
            displs[r] = offset;
            offset += counts[r];
        }
    }

    // --- Scatter seeds: root distributes contiguous cell-blocks (no Bcast) ---
    std::vector<int> local_seed(local_cells);
    t.comm_begin();
    MPI_Scatterv(rank == 0 ? seeds.data() : nullptr,
                 counts.data(), displs.data(), MPI_INT,
                 local_seed.data(), local_cells, MPI_INT,
                 0, MPI_COMM_WORLD);
    t.comm_end();

    // --- Compute local prices via SYCL (one work-item per local cell) ---
    std::vector<double> local_price(local_cells, 0.0);
    if (local_cells > 0) {
        dacpp_sycl::queue q{dacpp_sycl::default_selector_v};

        dacpp_sycl::buffer<int, 1> bufSeed(local_seed.data(),
                                           dacpp_sycl::range<1>(local_cells));
        dacpp_sycl::buffer<double, 1> bufPrice(local_price.data(),
                                               dacpp_sycl::range<1>(local_cells));

        t.comp_begin();
        q.submit([&](dacpp_sycl::handler& h) {
            auto s = bufSeed.get_access<dacpp_sycl::access::mode::read>(h);
            auto p = bufPrice.get_access<dacpp_sycl::access::mode::write>(h);
            h.parallel_for<class MonteCarloOptionKernel>(
                dacpp_sycl::range<1>(local_cells),
                MonteCarloOptionOp<decltype(s), decltype(p)>{s, p});
        });
        q.wait();
        t.comp_end();
    }

    // --- Gather prices to root (MPI_Gatherv, units = cells) ---
    std::vector<double> prices;
    if (rank == 0) prices.resize(total_cells);
    t.comm_begin();
    MPI_Gatherv(local_price.data(), local_cells, MPI_DOUBLE,
                rank == 0 ? prices.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    t.comm_end();

    if (rank == 0) {
        std::cout << prices[(NY / 2) * NX + (NX / 2)] << std::endl;
    }

    t.report();

    MPI_Finalize();
    return 0;
}

// Mandelbrot set — MPI + SYCL hand-written Tier-1 baseline (ALIGNED).
//
// This is a FAIR baseline: it uses the EXACT SAME parallel algorithm
// (data decomposition + communication pattern) as the DACPP-generated code
// (example/mandel/mandel.dac_sycl_buffer.cpp), written cleanly by hand.
//
// Alignment with DACPP generated "Contiguous1D" layout:
//   - complex_points is split by index contiguously across ranks
//     (Contiguous1D) and SCATTERED ONCE from root via MPI_Scatterv
//     (MPI_C_FLOAT_COMPLEX) — mirroring the generated single MANDEL_mandel
//     scatter of complex_points. There is no time loop.
//   - Each rank computes the Mandelbrot escape-iteration result for its owned
//     points; mandelbrot_flags[i]=1 iff the point stays bounded for
//     max_iterations — mirroring mandel_mpi_local().
//   - mandelbrot_flags is GATHERED to root via MPI_Gatherv (MPI_INT,
//     root-only output). Root counts/prints the set membership for sanity.
//
// Problem sizes and data initialization reuse the package baseline
// (programs/mandel1.0/mandel.MPI_StandardSycl.cpp): row_count=col_count=4096,
// max_iterations=1000, points laid out as complex<float>(real,imag).

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
#include <complex>
#include <cmath>
#include <iostream>
#include <vector>

#ifndef MANDEL_N
#define MANDEL_N 4096
#endif

static const int row_count = MANDEL_N;
static const int col_count = MANDEL_N;
static const int max_iterations = 1000;

// Mirrors mandel_mpi_local(): escape-iteration; flag set iff point bounded.
template <class PointsAcc, class FlagsAcc>
struct MandelAlignedOp {
    PointsAcc pts;
    FlagsAcc flags;

    void operator()(dacpp_sycl::id<1> idx) const {
        std::complex<float> c = pts[idx];
        std::complex<float> z(0.0f, 0.0f);
        int it = 0;
        for (int i = 0; i < max_iterations; ++i) {
            if (dacpp_sycl::sqrt(z.real() * z.real() + z.imag() * z.imag()) > 2.0f) {
                it = i;
                break;
            }
            z = std::complex<float>(
                z.real() * z.real() - z.imag() * z.imag() + c.real(),
                2.0f * z.real() * z.imag() + c.imag());
            it = max_iterations;
        }
        if (it == max_iterations) {
            flags[idx] = 1;
        }
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    BenchTimer timer("mandel baseline (aligned)");

    const int total_points = row_count * col_count;

    // Root builds the full complex_points grid; it is scattered once.
    std::vector<std::complex<float>> complex_points;
    if (rank == 0) {
        complex_points.resize(total_points);
        for (int i = 0; i < row_count; ++i) {
            for (int j = 0; j < col_count; ++j) {
                int index = i * col_count + j;
                float real = -1.5f + (i * (2.0f / row_count));
                float imag = -1.0f + (j * (2.0f / col_count));
                complex_points[index] = std::complex<float>(real, imag);
            }
        }
    }

    // Contiguous1D index split (base/rem) — same as matMul template.
    const int base = total_points / size;
    const int rem  = total_points % size;
    const int local_count  = base + (rank < rem ? 1 : 0);

    std::vector<int> counts(size), displs(size);
    {
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            counts[r] = base + (r < rem ? 1 : 0);
            displs[r] = offset;
            offset += counts[r];
        }
    }

    // Scatter complex_points contiguously (one-shot, aligned with generated).
    std::vector<std::complex<float>> local_points(local_count);
    timer.comm_begin();
    MPI_Scatterv(rank == 0 ? complex_points.data() : nullptr,
                 counts.data(), displs.data(), MPI_C_FLOAT_COMPLEX,
                 local_points.data(), local_count, MPI_C_FLOAT_COMPLEX,
                 0, MPI_COMM_WORLD);
    timer.comm_end();

    // Compute escape iterations for owned points.
    std::vector<int> local_flags(local_count, 0);
    if (local_count > 0) {
        timer.comp_begin();
        dacpp_sycl::queue q{dacpp_sycl::default_selector_v};
        dacpp_sycl::buffer<std::complex<float>, 1> buf_pts(
            local_points.data(), dacpp_sycl::range<1>(local_count));
        dacpp_sycl::buffer<int, 1> buf_flags(
            local_flags.data(), dacpp_sycl::range<1>(local_count));
        q.submit([&](dacpp_sycl::handler& h) {
            auto pts   = buf_pts.get_access<dacpp_sycl::access::mode::read>(h);
            auto flags = buf_flags.get_access<dacpp_sycl::access::mode::write>(h);
            h.parallel_for<class MandelAlignedKernel>(
                dacpp_sycl::range<1>(local_count),
                MandelAlignedOp<decltype(pts), decltype(flags)>{pts, flags});
        });
        q.wait();
        timer.comp_end();
    }

    // Gather flags to root (root-only output).
    std::vector<int> global_flags;
    if (rank == 0) global_flags.assign(total_points, 0);

    timer.comm_begin();
    MPI_Gatherv(local_flags.data(), local_count, MPI_INT,
                rank == 0 ? global_flags.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_INT, 0, MPI_COMM_WORLD);
    timer.comm_end();

    if (rank == 0) {
        int mandelbrot_count = 0;
        for (int f : global_flags) {
            if (f == 1) ++mandelbrot_count;
        }
        std::cout << "Mandelbrot Set Statistics:\n";
        std::cout << "Total points: " << total_points << "\n";
        std::cout << "Points in the Mandelbrot set: " << mandelbrot_count << "\n";
    }

    timer.report();

    MPI_Finalize();
    return 0;
}

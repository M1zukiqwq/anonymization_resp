// mandel perf/correctness source for validating the contiguous-scatter opt.
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdio>
#include <mpi.h>
#include "ReconTensor.h"
static inline bool __dacpp_mpi_is_root_rank();
using namespace std;
namespace dacpp { typedef std::vector<std::any> list; }

const int row_count = 8192, col_count = 8192, max_iterations = 1000;
vector<complex<float>> complex_points;
vector<int> mandelbrot_flags;
int total_points = 0;
long long mandelbrot_count = 0;

void InitializeComplexPoints() {
    total_points = row_count * col_count;
    complex_points.resize(total_points);
    for (int i = 0; i < row_count; ++i)
        for (int j = 0; j < col_count; ++j) {
            int index = i * col_count + j;
            float real = -1.5f + (i * (2.0f / row_count));
            float imag = -1.0f + (j * (2.0f / col_count));
            complex_points[index] = complex<float>(real, imag);
        }
}





static void __dacpp_e2e(const char* name, double secs) {
    double mx = 0.0, sm = 0.0;
    MPI_Reduce(&secs, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&secs, &sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    int r = 0, s = 1; MPI_Comm_rank(MPI_COMM_WORLD, &r); MPI_Comm_size(MPI_COMM_WORLD, &s);
    if (r == 0) std::printf("[MPI TEST] %s | ranks=%d | e2e_max=%.6f s | e2e_avg=%.6f s\n", name, s, mx, sm / s);
}

#include <sycl/sycl.hpp>
#include "DataReconstructor1.h"
#include "ParameterGeneration.h"
#include <mpi.h>
#include <cstdio>
#include "MPIPlanner.h"
#include <chrono>
#include <utility>

static inline bool __dacpp_mpi_is_root_rank() {
    int __dacpp_mpi_initialized = 0;
    MPI_Initialized(&__dacpp_mpi_initialized);
    if (!__dacpp_mpi_initialized) {
        return true;
    }
    int __dacpp_mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &__dacpp_mpi_rank);
    return __dacpp_mpi_rank == 0;
}

using namespace sycl;

template <typename __dacpp_view_t0, typename __dacpp_view_t1>
__attribute__((always_inline)) inline void mandel_mpi_local(__dacpp_view_t0 complex_points, __dacpp_view_t1 mandelbrot_flags) {
    const complex<float> &c = complex_points[0];
    complex<float> z = 0;
    int iterations = 0;
    for (int i = 0; i < max_iterations; ++i) {
        if (std::sqrt(z.real() * z.real() + z.imag() * z.imag()) > 2.F) {
            iterations = i;
            break;
        }
        z = z * z + c;
        iterations = max_iterations;
    }
    if (iterations == max_iterations)
        mandelbrot_flags[0] = 1;
}


void __dacpp_mpi_or_MANDEL_mandel_0(const dacpp::Vector<complex<float> > & __or_arg0, dacpp::Vector<int> & __or_arg1) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    dacpp::mpi::SegmentedProfile dacpp_profile;
    auto& q = dacpp::mpi::operator_resident::default_queue();
    auto dacpp_profile_init_start = dacpp::mpi::profileSegmentStart();
    const int64_t __or_total_items = __or_arg1.getShape(0);
    const auto __or_range = dacpp::mpi::operator_resident::rank_range_1d(__or_total_items, mpi_rank, mpi_size);
    const int64_t __or_local_item_count = __or_range.count;
    std::vector<int> __or_counts;
    std::vector<int> __or_displs;
    dacpp::mpi::operator_resident::counts_displs_1d(__or_total_items, mpi_size, __or_counts, __or_displs);
    auto __or_global_index_for_local = [=](int64_t __or_local_i) -> int64_t { return __or_range.begin + __or_local_i; };
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Init, dacpp_profile_init_start);
    std::vector<complex<float>> __or_local_complex_points(static_cast<std::size_t>(__or_local_item_count));
    std::vector<complex<float>> __or_global_complex_points;
    complex<float>* __or_scatter_src_complex_points = nullptr;
    bool __or_scatter_direct_complex_points = false;
    if (mpi_rank == 0) {
        const int64_t __or_direct_offset_complex_points = __or_arg0.getOffset();
        const int64_t __or_direct_stride_complex_points = __or_arg0.getStride(0);
        __or_scatter_direct_complex_points = (__or_direct_offset_complex_points >= 0 && __or_direct_stride_complex_points == 1 && __or_arg0.getSize() >= __or_total_items && __or_direct_offset_complex_points + __or_total_items <= __or_arg0.getSize());
        if (__or_scatter_direct_complex_points) {
            __or_scatter_src_complex_points = __or_arg0.getDataPtr().get() + __or_direct_offset_complex_points;
        }
    }
    if (mpi_rank == 0) {
        if (!__or_scatter_direct_complex_points) {
        auto dacpp_profile_pack_start_complex_points = dacpp::mpi::profileSegmentStart();
        __or_arg0.tensor2Array(__or_global_complex_points);
        __or_scatter_src_complex_points = __or_global_complex_points.data();
        dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Pack, dacpp_profile_pack_start_complex_points);
        }
    }
    auto dacpp_profile_scatter_start_complex_points = dacpp::mpi::profileSegmentStart();
    MPI_Scatterv(mpi_rank == 0 ? __or_scatter_src_complex_points : nullptr, mpi_rank == 0 ? __or_counts.data() : nullptr, mpi_rank == 0 ? __or_displs.data() : nullptr, MPI_C_FLOAT_COMPLEX, __or_local_complex_points.data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_local_item_count), "[DACPP][MPI][OR] scatter count exceeds MPI int range"), MPI_C_FLOAT_COMPLEX, 0, MPI_COMM_WORLD);
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Scatter, dacpp_profile_scatter_start_complex_points);
    std::vector<int> __or_local_mandelbrot_flags(static_cast<std::size_t>(__or_local_item_count));
    std::fill(__or_local_mandelbrot_flags.begin(), __or_local_mandelbrot_flags.end(), int{});
    // Output-direct no-read fast path for mandelbrot_flags initializes local output and skips root pack/scatter.
    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();
    if (__or_local_item_count > 0) {
        {
            sycl::buffer<complex<float>, 1> __or_buffer_complex_points(__or_local_complex_points.data(), sycl::range<1>(__or_local_complex_points.size()));
            sycl::buffer<int, 1> __or_buffer_mandelbrot_flags(__or_local_mandelbrot_flags.data(), sycl::range<1>(__or_local_mandelbrot_flags.size()));
            q.submit([&](sycl::handler& h) {
                auto __or_acc_complex_points = __or_buffer_complex_points.get_access<sycl::access::mode::read>(h);
                auto __or_acc_mandelbrot_flags = __or_buffer_mandelbrot_flags.get_access<sycl::access::mode::discard_write>(h);
                h.parallel_for<class __dacpp_mo_k0>(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* __or_data_complex_points = __or_acc_complex_points.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<const complex<float>> view_complex_points{__or_data_complex_points, item_linear};
                    auto* __or_data_mandelbrot_flags = __or_acc_mandelbrot_flags.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<int> view_mandelbrot_flags{__or_data_mandelbrot_flags, item_linear};
                    mandel_mpi_local(view_complex_points, view_mandelbrot_flags);
                });
            });
            q.wait();
        }
    }
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);
    std::vector<int> __or_materialized_mandelbrot_flags;
    dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_total_items), "[DACPP][MPI][OR] materialized output size exceeds MPI int range");
    if (mpi_rank == 0) {
        __or_materialized_mandelbrot_flags.resize(static_cast<std::size_t>(__or_total_items));
    }
    auto dacpp_profile_gather_start_mandelbrot_flags = dacpp::mpi::profileSegmentStart();
    MPI_Gatherv(__or_local_mandelbrot_flags.data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_local_item_count), "[DACPP][MPI][OR] gather count exceeds MPI int range"), MPI_INT, mpi_rank == 0 ? __or_materialized_mandelbrot_flags.data() : nullptr, mpi_rank == 0 ? __or_counts.data() : nullptr, mpi_rank == 0 ? __or_displs.data() : nullptr, MPI_INT, 0, MPI_COMM_WORLD);
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Gather, dacpp_profile_gather_start_mandelbrot_flags);
    auto dacpp_profile_materialize_start_mandelbrot_flags = dacpp::mpi::profileSegmentStart();
    if (mpi_rank == 0) {
        __or_arg1.array2Tensor(__or_materialized_mandelbrot_flags);
        dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_arg1.getSize()), "[DACPP][MPI][OR] materialized output broadcast count exceeds MPI int range");
        if (!__or_materialized_mandelbrot_flags.empty()) {
            auto dacpp_profile_bcast_start_mandelbrot_flags = dacpp::mpi::profileSegmentStart();
            MPI_Bcast(__or_materialized_mandelbrot_flags.data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_arg1.getSize()), "[DACPP][MPI][OR] materialized output broadcast count exceeds MPI int range"), MPI_INT, 0, MPI_COMM_WORLD);
            dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start_mandelbrot_flags);
        }
    } else {
        dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_arg1.getSize()), "[DACPP][MPI][OR] materialized output broadcast count exceeds MPI int range");
        __or_materialized_mandelbrot_flags.resize(static_cast<std::size_t>(__or_arg1.getSize()));
        if (!__or_materialized_mandelbrot_flags.empty()) {
            auto dacpp_profile_bcast_start_mandelbrot_flags = dacpp::mpi::profileSegmentStart();
            MPI_Bcast(__or_materialized_mandelbrot_flags.data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_arg1.getSize()), "[DACPP][MPI][OR] materialized output broadcast count exceeds MPI int range"), MPI_INT, 0, MPI_COMM_WORLD);
            dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Bcast, dacpp_profile_bcast_start_mandelbrot_flags);
        }
        __or_arg1.array2Tensor(__or_materialized_mandelbrot_flags);
    }
    dacpp::mpi::recordProfileSegment(dacpp_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start_mandelbrot_flags);
    // No downstream resident reader for mandelbrot_flags; host materialization above preserves visibility.
    dacpp::mpi::reportSegmentedProfile("__dacpp_mpi_or_MANDEL_mandel_0", dacpp_profile, MPI_COMM_WORLD);
}

int main() {
    int dacpp_mpi_finalize_needed = 0;
    int dacpp_mpi_initialized = 0;
    MPI_Initialized(&dacpp_mpi_initialized);
    if (!dacpp_mpi_initialized) {
        int dacpp_mpi_argc = 0;
        char** dacpp_mpi_argv = nullptr;
        MPI_Init(&dacpp_mpi_argc, &dacpp_mpi_argv);
        dacpp_mpi_finalize_needed = 1;
    }
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    InitializeComplexPoints();
    mandelbrot_flags.resize(total_points, 0);
    dacpp::Vector<complex<float>> complex_points_tensor(complex_points);
    dacpp::Vector<int> mandelbrot_flags_tensor(mandelbrot_flags);

    double __t0 = MPI_Wtime();
    __dacpp_mpi_or_MANDEL_mandel_0(complex_points_tensor, mandelbrot_flags_tensor);
    double __el = MPI_Wtime() - __t0;

    mandelbrot_flags_tensor.tensor2Array(mandelbrot_flags);
    long long cnt = 0; for (int f : mandelbrot_flags) cnt += f;
    int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r);
    if (r == 0) std::printf("[CHECK] mandel bounded_count=%lld total=%d\n", cnt, total_points);
    __dacpp_e2e("mandel.opt", __el);
    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
        dacpp_mpi_finalize_needed = 0;
    }
return 0;
}

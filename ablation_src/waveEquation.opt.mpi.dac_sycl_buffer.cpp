// waveEquation perf/correctness source (1024^2, 500 steps).
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <mpi.h>
#include "ReconTensor.h"
static inline bool __dacpp_mpi_is_root_rank();
using namespace std;
namespace dacpp { typedef std::vector<std::any> list; }

const int NX = 1024;
const int NY = 1024;
const double Lx = 10.0, Ly = 10.0, c = 1.0;
const int TIME_STEPS = 500;
const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);
const double dt = 0.5 * std::fmin(dx, dy) / c;





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

template <typename __dacpp_view_t0, typename __dacpp_view_t1, typename __dacpp_view_t2>
__attribute__((always_inline)) inline void waveEq_mpi_local(__dacpp_view_t0 cur, __dacpp_view_t1 prev, __dacpp_view_t2 next) {
    double dt = 0.5 * std::fmin(dx, dy) / c;
    double u_xx = (cur[2][1] - 2. * cur[1][1] + cur[0][1]) / (dx * dx);
    double u_yy = (cur[1][2] - 2. * cur[1][1] + cur[1][0]) / (dy * dy);
    next[0] = 2. * cur[1][1] - prev[0] + (c * c) * dt * dt * (u_xx + u_yy);
}


struct __dacpp_mpi_or_waveEqShell_waveEq_0_ctx {
    int mpi_rank = 0;
    int mpi_size = 1;
    int64_t __or_input_rows = 0;
    int64_t __or_input_cols = 0;
    int64_t __or_output_rows = 0;
    int64_t __or_output_cols = 0;
    int64_t __or_local_output_rows = 0;
    int64_t __or_local_output_cols = 0;
    int64_t __or_output_row_begin = 0;
    int64_t __or_output_col_begin = 0;
    int64_t __or_local_item_count = 0;
    int __or_window_rows = 3;
    int __or_window_cols = 3;
    int __or_followup_row_offset = 1;
    int __or_followup_col_offset = 1;
    int __or_temporal_block_size = 2;
    dacpp::mpi::operator_resident::RankRange1D __or_output_row_range{};
    dacpp::mpi::operator_resident::RankRange1D __or_output_col_range{};
    dacpp::mpi::operator_resident::ResidentHalo2DRowLayout __or_halo_layout{};
    dacpp::mpi::operator_resident::ResidentHalo2DSpatialLayout __or_spatial_layout{};
    std::vector<int> __or_row_counts;
    std::vector<int> __or_row_displs;
    std::vector<int> __or_counts;
    std::vector<int> __or_displs;
    dacpp::mpi::SegmentedProfile __or_profile;
    sycl::queue& q = dacpp::mpi::operator_resident::default_queue();
    std::vector<double> __or_local_cur;
    std::vector<double> __or_local_prev;
    std::vector<double> __or_local_next;
};
void __dacpp_mpi_or_waveEqShell_waveEq_0_init(__dacpp_mpi_or_waveEqShell_waveEq_0_ctx& ctx, dacpp::Matrix<double> & __or_arg0, dacpp::Matrix<double> & __or_arg1, dacpp::Matrix<double> & __or_arg2) {
    auto dacpp_profile_init_start = dacpp::mpi::profileSegmentStart();
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);
    ctx.__or_input_rows = __or_arg0.getShape(0);
    ctx.__or_input_cols = __or_arg0.getShape(1);
    ctx.__or_output_rows = __or_arg2.getShape(0);
    ctx.__or_output_cols = __or_arg2.getShape(1);
    ctx.__or_output_row_range = dacpp::mpi::operator_resident::rank_range_1d(ctx.__or_output_rows, ctx.mpi_rank, ctx.mpi_size);
    ctx.__or_output_col_range = {0, ctx.__or_output_cols};
    ctx.__or_local_output_rows = ctx.__or_output_row_range.count;
    ctx.__or_local_output_cols = ctx.__or_output_cols;
    ctx.__or_output_row_begin = ctx.__or_output_row_range.begin;
    ctx.__or_output_col_begin = 0;
    ctx.__or_local_item_count = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(ctx.__or_local_output_rows), static_cast<int64_t>(ctx.__or_local_output_cols), "[DACPP][MPI][OR] resident halo 2D local item count overflow");
    ctx.__or_halo_layout = dacpp::mpi::operator_resident::resident_halo_2d_row_layout_temporal(ctx.__or_output_rows, ctx.__or_input_cols, ctx.mpi_rank, ctx.mpi_size, ctx.__or_temporal_block_size);
    dacpp::mpi::operator_resident::counts_displs_1d(ctx.__or_output_rows, ctx.mpi_size, ctx.__or_row_counts, ctx.__or_row_displs);
    ctx.__or_counts.resize(static_cast<std::size_t>(ctx.mpi_size));
    ctx.__or_displs.resize(static_cast<std::size_t>(ctx.mpi_size));
    for (int r = 0; r < ctx.mpi_size; ++r) {
        ctx.__or_counts[static_cast<std::size_t>(r)] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(ctx.__or_row_counts[static_cast<std::size_t>(r)]), static_cast<int64_t>(ctx.__or_output_cols), "[DACPP][MPI][OR] resident halo 2D gather count exceeds MPI int range")), "[DACPP][MPI][OR] resident halo 2D gather count exceeds MPI int range");
        ctx.__or_displs[static_cast<std::size_t>(r)] = dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(ctx.__or_row_displs[static_cast<std::size_t>(r)]), static_cast<int64_t>(ctx.__or_output_cols), "[DACPP][MPI][OR] resident halo 2D gather displacement exceeds MPI int range")), "[DACPP][MPI][OR] resident halo 2D gather displacement exceeds MPI int range");
    }
    dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(ctx.__or_halo_layout.local_size), "[DACPP][MPI][OR] resident halo 2D local reader size exceeds MPI int range");
    dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(ctx.__or_local_item_count), "[DACPP][MPI][OR] resident halo 2D local writer size exceeds MPI int range");
    const int64_t __or_local_reader_size = ctx.__or_halo_layout.local_size;
    ctx.__or_local_cur.assign(static_cast<std::size_t>(__or_local_reader_size), double{});
    ctx.__or_local_prev.assign(static_cast<std::size_t>(__or_local_reader_size), double{});
    ctx.__or_local_next.assign(static_cast<std::size_t>(__or_local_reader_size), double{});
    const int64_t __or_initial_reader_dense_count = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(ctx.__or_input_rows), static_cast<int64_t>(ctx.__or_input_cols), "[DACPP][MPI][OR] resident halo 2D initial reader size overflow");
    dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_initial_reader_dense_count), "[DACPP][MPI][OR] resident halo 2D initial reader size exceeds MPI int range");
    std::vector<double> __or_initial_global_cur;
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Init, dacpp_profile_init_start);
    auto dacpp_profile_scatter_start = dacpp::mpi::profileSegmentStart();
    std::fill(ctx.__or_local_cur.begin(), ctx.__or_local_cur.end(), 0.0);
    // Constant-initialized resident halo reader cur is filled locally; skip root tensor2Array/scatter_window_2d_rows.
    const int64_t __or_direct_rows_prev = __or_arg1.getShape(0);
    const int64_t __or_direct_cols_prev = __or_arg1.getShape(1);
    if (__or_direct_rows_prev != ctx.__or_output_rows || __or_direct_cols_prev != ctx.__or_output_cols) {
        if (ctx.mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR] resident halo direct reader matPrev shape mismatch with output\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    std::vector<double> __or_initial_global_prev;
    if (ctx.mpi_rank == 0) {
        __or_arg1.tensor2Array(__or_initial_global_prev);
    }
    dacpp::mpi::operator_resident::scatter_window_2d_rows_temporal_offset(__or_initial_global_prev, ctx.__or_local_prev, ctx.__or_output_rows, ctx.__or_output_cols, ctx.__or_input_cols, ctx.__or_temporal_block_size, ctx.__or_followup_row_offset, ctx.__or_followup_col_offset, ctx.__or_halo_layout, ctx.mpi_rank, ctx.mpi_size, MPI_DOUBLE);
    // Direct-reader recurrence state is packed straight into the temporal resident halo layout.
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Scatter, dacpp_profile_scatter_start);
}
void __dacpp_mpi_or_waveEqShell_waveEq_0_run(__dacpp_mpi_or_waveEqShell_waveEq_0_ctx& ctx, dacpp::Matrix<double> & __or_arg0, dacpp::Matrix<double> & __or_arg1, dacpp::Matrix<double> & __or_arg2) {
    int mpi_rank = ctx.mpi_rank;
    auto& q = ctx.q;
    const int64_t __or_input_rows = ctx.__or_input_rows;
    const int64_t __or_input_cols = ctx.__or_input_cols;
    const int64_t __or_output_rows = ctx.__or_output_rows;
    const int64_t __or_output_cols = ctx.__or_output_cols;
    const int64_t __or_local_output_rows = ctx.__or_local_output_rows;
    const int64_t __or_local_output_cols = ctx.__or_local_output_cols;
    const int64_t __or_local_item_count = ctx.__or_local_item_count;
    const int64_t __or_local_reader_rows = ctx.__or_halo_layout.local_row_count;
    const int64_t __or_local_reader_cols = __or_input_cols;
    const int64_t __or_local_reader_row_begin = ctx.__or_halo_layout.global_row_begin;
    const int64_t __or_local_reader_col_begin = 0;
    const int __or_writer_row_offset = ctx.__or_followup_row_offset;
    const int __or_writer_col_offset = ctx.__or_followup_col_offset;
    int64_t __or_kernel_item_count = __or_local_item_count;
    const int __or_temporal_block_size = ctx.__or_temporal_block_size;
    const bool __or_temporal_block_safe = __or_output_rows >= static_cast<int64_t>(ctx.mpi_size) * static_cast<int64_t>(__or_temporal_block_size);
    const int __or_runtime_temporal_block_size = __or_temporal_block_safe ? __or_temporal_block_size : 1;
    auto& __or_local_cur = ctx.__or_local_cur;
    auto& __or_local_prev = ctx.__or_local_prev;
    auto& __or_local_next = ctx.__or_local_next;
    int64_t __or_time_step = 0;
    const int64_t __or_time_limit = static_cast<int64_t>(TIME_STEPS);
    const int64_t __or_time_end = __or_time_limit;
    while (__or_time_step < __or_time_end) {
    const int __or_inner_steps = static_cast<int>(std::min<int64_t>(__or_runtime_temporal_block_size, __or_time_end - __or_time_step));
    auto dacpp_profile_halo_start = dacpp::mpi::profileSegmentStart();
    dacpp::mpi::operator_resident::exchange_halo_2d_rows_temporal_inplace(__or_local_cur, ctx.__or_halo_layout, __or_output_rows, __or_output_cols, __or_input_cols, ctx.__or_followup_row_offset, ctx.__or_followup_col_offset, __or_temporal_block_size, mpi_rank, ctx.mpi_size, MPI_DOUBLE);
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Halo, dacpp_profile_halo_start);
    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();
    for (int __or_block_step = 0; __or_block_step < __or_inner_steps; ++__or_block_step) {
        const int64_t __or_compute_row_begin = std::max<int64_t>(1, ctx.__or_halo_layout.owned_row_offset + 1 - (__or_inner_steps - __or_block_step - 1));
        const int64_t __or_compute_row_end = std::min<int64_t>(__or_local_reader_rows - 1, ctx.__or_halo_layout.owned_row_offset + 1 + __or_local_output_rows + (__or_inner_steps - __or_block_step - 1));
        const int64_t __or_compute_rows = std::max<int64_t>(0, __or_compute_row_end - __or_compute_row_begin);
        __or_kernel_item_count = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(__or_compute_rows, __or_local_output_cols, "[DACPP][MPI][OR] temporal resident halo 2D kernel item count overflow");
        if (__or_kernel_item_count > 0) {
            sycl::buffer<double, 1> __or_reader_buf(__or_local_cur.data(), sycl::range<1>(__or_local_cur.size()));
            sycl::buffer<double, 1> __or_direct_reader_buf(__or_local_prev.data(), sycl::range<1>(__or_local_prev.size()));
            sycl::buffer<double, 1> __or_writer_buf(__or_local_next.data(), sycl::range<1>(__or_local_next.size()));
            q.submit([&](sycl::handler& h) {
                auto __or_reader_acc = __or_reader_buf.get_access<sycl::access::mode::read>(h);
                auto __or_direct_reader_acc = __or_direct_reader_buf.get_access<sycl::access::mode::read>(h);
                auto __or_writer_acc = __or_writer_buf.get_access<sycl::access::mode::read_write>(h);
                h.parallel_for<class __dacpp_wv_k0>(sycl::range<1>(static_cast<std::size_t>(__or_kernel_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    const int local_row = item_linear / static_cast<int>(__or_local_output_cols);
                    const int local_col = item_linear % static_cast<int>(__or_local_output_cols);
                    auto* __or_reader_data = __or_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* __or_direct_reader_data = __or_direct_reader_acc.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* __or_writer_data = __or_writer_acc.template get_multi_ptr<sycl::access::decorated::no>().get();
                    const int __or_reader_base_idx = static_cast<int>((local_row + __or_compute_row_begin - 1) * static_cast<int>(__or_local_reader_cols) + local_col + static_cast<int>(0));
                    auto* __or_reader_base = __or_reader_data + __or_reader_base_idx;
                    const int __or_writer_base_idx = static_cast<int>((local_row + __or_compute_row_begin) * static_cast<int>(__or_local_reader_cols) + local_col + static_cast<int>(__or_writer_col_offset));
                    auto* __or_writer_base = __or_writer_data + __or_writer_base_idx;
                    auto* __or_direct_reader_base = __or_direct_reader_data + __or_writer_base_idx;
                    dacpp::mpi::ResidentHaloView2D<const double> view_cur{__or_reader_base, 0, static_cast<int>(__or_local_reader_cols)};
                    dacpp::mpi::ContiguousView1D<const double> view_prev{__or_direct_reader_base, 0};
                    dacpp::mpi::ContiguousView1D<double> view_next{__or_writer_base, 0};
                    waveEq_mpi_local(view_cur, view_prev, view_next);
                });
            });
            q.wait();
        }
        std::swap(__or_local_prev, __or_local_cur);
        std::swap(__or_local_cur, __or_local_next);
    {
        const int64_t __or_boundary_col = static_cast<int64_t>(NY - 1);
        if (__or_boundary_col >= __or_local_reader_col_begin && __or_boundary_col < __or_local_reader_col_begin + __or_local_reader_cols) {
            const int64_t __or_boundary_begin = std::max<int64_t>(static_cast<int64_t>(0), __or_local_reader_row_begin);
            const int64_t __or_boundary_end = std::min<int64_t>((static_cast<int64_t>(NX - 1) + 1), __or_local_reader_row_begin + __or_local_reader_rows);
            const int64_t __or_boundary_local_col = __or_boundary_col - __or_local_reader_col_begin;
            for (int64_t __or_boundary_row = __or_boundary_begin; __or_boundary_row < __or_boundary_end; ++__or_boundary_row) {
                const int64_t __or_boundary_target_local = (__or_boundary_row - __or_local_reader_row_begin) * __or_local_reader_cols + __or_boundary_local_col;
                if (__or_boundary_target_local < 0 || static_cast<std::size_t>(__or_boundary_target_local) >= __or_local_cur.size()) continue;
                __or_local_cur[static_cast<std::size_t>(__or_boundary_target_local)] = static_cast<double>(0);
            }
        }
    }
    {
        const int64_t __or_boundary_col = static_cast<int64_t>(0);
        if (__or_boundary_col >= __or_local_reader_col_begin && __or_boundary_col < __or_local_reader_col_begin + __or_local_reader_cols) {
            const int64_t __or_boundary_begin = std::max<int64_t>(static_cast<int64_t>(0), __or_local_reader_row_begin);
            const int64_t __or_boundary_end = std::min<int64_t>((static_cast<int64_t>(NX - 1) + 1), __or_local_reader_row_begin + __or_local_reader_rows);
            const int64_t __or_boundary_local_col = __or_boundary_col - __or_local_reader_col_begin;
            for (int64_t __or_boundary_row = __or_boundary_begin; __or_boundary_row < __or_boundary_end; ++__or_boundary_row) {
                const int64_t __or_boundary_target_local = (__or_boundary_row - __or_local_reader_row_begin) * __or_local_reader_cols + __or_boundary_local_col;
                if (__or_boundary_target_local < 0 || static_cast<std::size_t>(__or_boundary_target_local) >= __or_local_cur.size()) continue;
                __or_local_cur[static_cast<std::size_t>(__or_boundary_target_local)] = static_cast<double>(0);
            }
        }
    }
    {
        const int64_t __or_boundary_row = static_cast<int64_t>(NX - 1);
        if (__or_boundary_row >= __or_local_reader_row_begin && __or_boundary_row < __or_local_reader_row_begin + __or_local_reader_rows) {
            const int64_t __or_boundary_begin = std::max<int64_t>(static_cast<int64_t>(0), __or_local_reader_col_begin);
            const int64_t __or_boundary_end = std::min<int64_t>((static_cast<int64_t>(NY - 1) + 1), __or_local_reader_col_begin + __or_local_reader_cols);
            const int64_t __or_boundary_local_row = __or_boundary_row - __or_local_reader_row_begin;
            for (int64_t __or_boundary_col = __or_boundary_begin; __or_boundary_col < __or_boundary_end; ++__or_boundary_col) {
                const int64_t __or_boundary_target_local = __or_boundary_local_row * __or_local_reader_cols + (__or_boundary_col - __or_local_reader_col_begin);
                if (__or_boundary_target_local < 0 || static_cast<std::size_t>(__or_boundary_target_local) >= __or_local_cur.size()) continue;
                __or_local_cur[static_cast<std::size_t>(__or_boundary_target_local)] = static_cast<double>(0);
            }
        }
    }
    {
        const int64_t __or_boundary_row = static_cast<int64_t>(0);
        if (__or_boundary_row >= __or_local_reader_row_begin && __or_boundary_row < __or_local_reader_row_begin + __or_local_reader_rows) {
            const int64_t __or_boundary_begin = std::max<int64_t>(static_cast<int64_t>(0), __or_local_reader_col_begin);
            const int64_t __or_boundary_end = std::min<int64_t>((static_cast<int64_t>(NY - 1) + 1), __or_local_reader_col_begin + __or_local_reader_cols);
            const int64_t __or_boundary_local_row = __or_boundary_row - __or_local_reader_row_begin;
            for (int64_t __or_boundary_col = __or_boundary_begin; __or_boundary_col < __or_boundary_end; ++__or_boundary_col) {
                const int64_t __or_boundary_target_local = __or_boundary_local_row * __or_local_reader_cols + (__or_boundary_col - __or_local_reader_col_begin);
                if (__or_boundary_target_local < 0 || static_cast<std::size_t>(__or_boundary_target_local) >= __or_local_cur.size()) continue;
                __or_local_cur[static_cast<std::size_t>(__or_boundary_target_local)] = static_cast<double>(0);
            }
        }
    }
    }
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);
    __or_time_step += __or_inner_steps;
    }
    // P4.6 temporal-block=2 leaves the latest state in the resident reader buffer after the block loop.
}
void __dacpp_mpi_or_waveEqShell_waveEq_0_materialize(__dacpp_mpi_or_waveEqShell_waveEq_0_ctx& ctx, dacpp::Matrix<double> & __or_arg0, dacpp::Matrix<double> & __or_arg1, dacpp::Matrix<double> & __or_arg2) {
    int mpi_rank = ctx.mpi_rank;
    std::vector<double> __or_materialized_next;
    const int64_t __or_materialized_writer_count = dacpp::mpi::operator_resident::checked_mul_int64_or_abort(static_cast<int64_t>(ctx.__or_output_rows), static_cast<int64_t>(ctx.__or_output_cols), "[DACPP][MPI][OR] resident halo 2D materialized output size overflow");
    auto dacpp_profile_gather_start_writer = dacpp::mpi::profileSegmentStart();
    const auto __or_owned_next = dacpp::mpi::operator_resident::owned_slice_2d_rows_temporal(ctx.__or_local_cur, ctx.__or_halo_layout, ctx.__or_output_cols, ctx.__or_input_cols, 1, 1);
    dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(__or_materialized_writer_count), "[DACPP][MPI][OR] resident halo 2D materialized output size exceeds MPI int range");
    if (mpi_rank == 0) {
        __or_materialized_next.resize(static_cast<std::size_t>(__or_materialized_writer_count));
    }
    MPI_Gatherv(__or_owned_next.data(), dacpp::mpi::operator_resident::narrow_mpi_count_or_abort(static_cast<int64_t>(ctx.__or_local_item_count), "[DACPP][MPI][OR] resident halo 2D materialize sendcount exceeds MPI int range"), MPI_DOUBLE, mpi_rank == 0 ? __or_materialized_next.data() : nullptr, mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Gather, dacpp_profile_gather_start_writer);
    auto dacpp_profile_materialize_start = dacpp::mpi::profileSegmentStart();
    // No host post-use for next; skip full resident halo writer materialization.
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start);
    // No host post-use for prev; skip full resident halo direct-reader materialization.
    dacpp_profile_materialize_start = dacpp::mpi::profileSegmentStart();
    if (mpi_rank == 0) {
        std::vector<double> __or_materialized_cur;
        __or_arg0.tensor2Array(__or_materialized_cur);
        const int64_t __or_followup_reader_cols_cur = static_cast<int64_t>(__or_arg0.getShape(1));
        for (std::size_t __or_idx = 0; __or_idx < __or_materialized_next.size(); ++__or_idx) {
            const int64_t __or_target = dacpp::mpi::map_2d_global_with_offset(static_cast<int64_t>(__or_idx), ctx.__or_output_cols, __or_followup_reader_cols_cur, 1, 1);
            if (__or_target >= 0 && static_cast<std::size_t>(__or_target) < __or_materialized_cur.size()) {
                __or_materialized_cur[static_cast<std::size_t>(__or_target)] = static_cast<double>(__or_materialized_next[__or_idx]);
            }
        }
            {
                const int64_t __or_boundary_begin = static_cast<int64_t>(0);
                const int64_t __or_boundary_end = static_cast<int64_t>(NX - 1);
                for (int64_t __or_boundary_idx = __or_boundary_begin; __or_boundary_idx <= __or_boundary_end; ++__or_boundary_idx) {
                    const int64_t __or_boundary_target_row = __or_boundary_idx;
                    const int64_t __or_boundary_target_col = NY - 1;
                    if (__or_boundary_target_row < 0 || __or_boundary_target_col < 0 || __or_boundary_target_row >= __or_arg0.getShape(0) || __or_boundary_target_col >= __or_followup_reader_cols_cur) continue;
                    const int64_t __or_boundary_target = __or_boundary_target_row * __or_followup_reader_cols_cur + __or_boundary_target_col;
                    if (__or_boundary_target < 0 || static_cast<std::size_t>(__or_boundary_target) >= __or_materialized_cur.size()) continue;
                    __or_materialized_cur[static_cast<std::size_t>(__or_boundary_target)] = static_cast<double>(0);
                }
            }
            {
                const int64_t __or_boundary_begin = static_cast<int64_t>(0);
                const int64_t __or_boundary_end = static_cast<int64_t>(NX - 1);
                for (int64_t __or_boundary_idx = __or_boundary_begin; __or_boundary_idx <= __or_boundary_end; ++__or_boundary_idx) {
                    const int64_t __or_boundary_target_row = __or_boundary_idx;
                    const int64_t __or_boundary_target_col = 0;
                    if (__or_boundary_target_row < 0 || __or_boundary_target_col < 0 || __or_boundary_target_row >= __or_arg0.getShape(0) || __or_boundary_target_col >= __or_followup_reader_cols_cur) continue;
                    const int64_t __or_boundary_target = __or_boundary_target_row * __or_followup_reader_cols_cur + __or_boundary_target_col;
                    if (__or_boundary_target < 0 || static_cast<std::size_t>(__or_boundary_target) >= __or_materialized_cur.size()) continue;
                    __or_materialized_cur[static_cast<std::size_t>(__or_boundary_target)] = static_cast<double>(0);
                }
            }
            {
                const int64_t __or_boundary_begin = static_cast<int64_t>(0);
                const int64_t __or_boundary_end = static_cast<int64_t>(NY - 1);
                for (int64_t __or_boundary_idx = __or_boundary_begin; __or_boundary_idx <= __or_boundary_end; ++__or_boundary_idx) {
                    const int64_t __or_boundary_target_row = NX - 1;
                    const int64_t __or_boundary_target_col = __or_boundary_idx;
                    if (__or_boundary_target_row < 0 || __or_boundary_target_col < 0 || __or_boundary_target_row >= __or_arg0.getShape(0) || __or_boundary_target_col >= __or_followup_reader_cols_cur) continue;
                    const int64_t __or_boundary_target = __or_boundary_target_row * __or_followup_reader_cols_cur + __or_boundary_target_col;
                    if (__or_boundary_target < 0 || static_cast<std::size_t>(__or_boundary_target) >= __or_materialized_cur.size()) continue;
                    __or_materialized_cur[static_cast<std::size_t>(__or_boundary_target)] = static_cast<double>(0);
                }
            }
            {
                const int64_t __or_boundary_begin = static_cast<int64_t>(0);
                const int64_t __or_boundary_end = static_cast<int64_t>(NY - 1);
                for (int64_t __or_boundary_idx = __or_boundary_begin; __or_boundary_idx <= __or_boundary_end; ++__or_boundary_idx) {
                    const int64_t __or_boundary_target_row = 0;
                    const int64_t __or_boundary_target_col = __or_boundary_idx;
                    if (__or_boundary_target_row < 0 || __or_boundary_target_col < 0 || __or_boundary_target_row >= __or_arg0.getShape(0) || __or_boundary_target_col >= __or_followup_reader_cols_cur) continue;
                    const int64_t __or_boundary_target = __or_boundary_target_row * __or_followup_reader_cols_cur + __or_boundary_target_col;
                    if (__or_boundary_target < 0 || static_cast<std::size_t>(__or_boundary_target) >= __or_materialized_cur.size()) continue;
                    __or_materialized_cur[static_cast<std::size_t>(__or_boundary_target)] = static_cast<double>(0);
                }
            }
        __or_arg0.array2Tensor(__or_materialized_cur);
    }
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start);
    (void)__or_arg1;
    dacpp::mpi::reportSegmentedProfile("__dacpp_mpi_or_waveEqShell_waveEq_0_materialize", ctx.__or_profile, MPI_COMM_WORLD);
}
void __dacpp_mpi_or_waveEqShell_waveEq_0(dacpp::Matrix<double> & __or_arg0, dacpp::Matrix<double> & __or_arg1, dacpp::Matrix<double> & __or_arg2) {
    __dacpp_mpi_or_waveEqShell_waveEq_0_ctx ctx;
    __dacpp_mpi_or_waveEqShell_waveEq_0_init(ctx, __or_arg0, __or_arg1, __or_arg2);
    __dacpp_mpi_or_waveEqShell_waveEq_0_run(ctx, __or_arg0, __or_arg1, __or_arg2);
    __dacpp_mpi_or_waveEqShell_waveEq_0_materialize(ctx, __or_arg0, __or_arg1, __or_arg2);
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

    vector<double> u_prev(NX * NY, 0.0), u_curr(NX * NY, 0.0), u_next(NX * NY, 0.0);
    double sigma = 0.5;
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j) {
            double x = i * dx, y = j * dy;
            u_prev[i * NX + j] = std::exp(-((x - Lx / 2) * (x - Lx / 2) + (y - Ly / 2) * (y - Ly / 2)) / (2 * sigma * sigma));
        }
    dacpp::Matrix<double> matCur({NX, NY}, u_curr);
    dacpp::Matrix<double> u_prev_tensor({NX, NY}, u_prev);
    dacpp::Matrix<double> u_next_tensor({NX, NY}, u_next);
    dacpp::Matrix<double> matPrev = u_prev_tensor[{1, NX - 1}][{1, NY - 1}];
    dacpp::Matrix<double> matNext = u_next_tensor[{1, NX - 1}][{1, NY - 1}];

    double __t0 = MPI_Wtime();
        __dacpp_mpi_or_waveEqShell_waveEq_0_ctx __dacpp_mpi_or_ctx_0;
    __dacpp_mpi_or_waveEqShell_waveEq_0_init(__dacpp_mpi_or_ctx_0, matCur, matPrev, matNext);
    __dacpp_mpi_or_waveEqShell_waveEq_0_run(__dacpp_mpi_or_ctx_0, matCur, matPrev, matNext);
    __dacpp_mpi_or_waveEqShell_waveEq_0_materialize(__dacpp_mpi_or_ctx_0, matCur, matPrev, matNext);
for (int t = 0; false; t++) {
        ((void)0);
        ;
        ;
        
        
    }
    double __el = MPI_Wtime() - __t0;

    int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r);
    if (r == 0) {
        double cs = 0.0; for (int j = 0; j < NY; ++j) cs += matCur[NX / 2][j];
        std::printf("[CHECK] wave midrow_checksum=%.6f\n", cs);
    }
    __dacpp_e2e("wave.opt", __el);
    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
        dacpp_mpi_finalize_needed = 0;
    }
return 0;
}

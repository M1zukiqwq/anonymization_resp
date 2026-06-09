// decay ablation benchmark source (DSL + e2e timing harness).
// Translate ON:  dacpp.sh translate ... --mpi   (operator-resident)
// Translate OFF: dacpp.sh translate ... --mpi --mpi-no-operator-resident (legacy gather)
// The translator injects MPI_Init/Finalize + the root-rank helper into main();
// we time the whole compute region and print the [MPI TEST] e2e line.
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <mpi.h>
#include "ReconTensor.h"
static inline bool __dacpp_mpi_is_root_rank();
namespace dacpp {
    typedef std::vector<std::any> list;
}

const double dt = 0.1;
const double T = 100.0;
const size_t numIsotopes = 1024576;





static void __dacpp_e2e(const char* name, double secs) {
    double mx = 0.0, sm = 0.0;
    MPI_Reduce(&secs, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&secs, &sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    int r = 0, s = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    if (r == 0)
        std::printf("[MPI TEST] %s | ranks=%d | e2e_max=%.6f s | e2e_avg=%.6f s\n",
                    name, s, mx, sm / s);
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

template <typename __dacpp_view_t0, typename __dacpp_view_t1, typename __dacpp_view_t2, typename __dacpp_view_t3>
__attribute__((always_inline)) inline void decay_mpi_local(__dacpp_view_t0 N0s, __dacpp_view_t1 lambdas, __dacpp_view_t2 local_A, __dacpp_view_t3 t) {
    local_A[0] = N0s[0] * std::exp(-lambdas[0] * t[0]);
}


struct __dacpp_mpi_stencil_ctx___dacpp_mpi_stencil_DECAY_decay_0 {
    int mpi_rank = 0;
    int mpi_size = 1;
    bool use_partial_exchange = false;
    bool use_contiguous_kernel_views = false;
    std::string partial_exchange_disable_reason;
    std::unique_ptr<sycl::queue> q;
    std::vector<int64_t> binding_split_sizes;
    int64_t total_items = 1;
    dacpp::mpi::ItemRange item_range{};
    int64_t local_item_count = 0;
    dacpp::mpi::WaveSpecializationState wave;
    dacpp::mpi::AccessPattern pattern_N0s;
    dacpp::mpi::PackPlan plan_N0s;
    std::vector<double> local_N0s;
    dacpp::mpi::DistributedTensorState<double> dist_N0s;
    dacpp::mpi::GatheredIndexLayout input_layout_N0s;
    std::vector<double> global_N0s;
    std::vector<double> sendbuf_N0s;
    dacpp::mpi::AccessPattern pattern_lambdas;
    dacpp::mpi::PackPlan plan_lambdas;
    std::vector<double> local_lambdas;
    dacpp::mpi::DistributedTensorState<double> dist_lambdas;
    dacpp::mpi::GatheredIndexLayout input_layout_lambdas;
    std::vector<double> global_lambdas;
    std::vector<double> sendbuf_lambdas;
    dacpp::mpi::AccessPattern pattern_local_A;
    dacpp::mpi::PackPlan plan_local_A;
    std::vector<double> local_local_A;
    dacpp::mpi::DistributedTensorState<double> dist_local_A;
    dacpp::mpi::GatheredIndexLayout output_layout_local_A;
    std::vector<int32_t> writeback_slots_local_A;
    std::vector<double> writeback_values_local_A;
    std::vector<double> global_recv_values_local_A;
    std::vector<double> global_out_local_A;
    dacpp::mpi::AccessPattern pattern_t;
    dacpp::mpi::PackPlan plan_t;
    std::vector<double> local_t;
    dacpp::mpi::DistributedTensorState<double> dist_t;
    dacpp::mpi::GatheredIndexLayout input_layout_t;
    std::vector<double> global_t;
    std::vector<double> sendbuf_t;
};

void __dacpp_mpi_stencil_init___dacpp_mpi_stencil_DECAY_decay_0(__dacpp_mpi_stencil_ctx___dacpp_mpi_stencil_DECAY_decay_0& ctx, const dacpp::Vector<double> & N0s, const dacpp::Vector<double> & lambdas, dacpp::Vector<double> & local_A, const dacpp::Vector<double> & t) {
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);
    ctx.q = std::make_unique<sycl::queue>(sycl::default_selector_v);
    ctx.binding_split_sizes.clear();
    ctx.total_items = 1;
    ctx.pattern_N0s = dacpp::mpi::AccessPattern{};
    ctx.pattern_N0s.param_id = 0;
    ctx.pattern_N0s.name = "N0s";
    ctx.pattern_N0s.mode = dacpp::mpi::AccessMode::Read;
    ctx.pattern_N0s.data_info.dim = N0s.getDim();
    for (int dim = 0; dim < N0s.getDim(); ++dim) ctx.pattern_N0s.data_info.dimLength.push_back(N0s.getShape(dim));
    Dac_Op pattern_N0s_op_0;
    pattern_N0s_op_0.setDimId(0);
    pattern_N0s_op_0.size = 1;
    pattern_N0s_op_0.stride = 1;
    pattern_N0s_op_0.SetSplitSize(N0s.getShape(0));
    ctx.pattern_N0s.param_ops.push_back(pattern_N0s_op_0);
    ctx.pattern_N0s.bind_set_id.push_back(0);
    ctx.pattern_N0s.bind_offset_expr.push_back("0");
    ctx.pattern_N0s.is_index_op.push_back(true);
    ctx.pattern_N0s.partition_shape = dacpp::mpi::init_partition_shape(ctx.pattern_N0s);
    ctx.pattern_N0s.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(ctx.pattern_N0s);
    if (ctx.binding_split_sizes.size() < ctx.pattern_N0s.bind_split_sizes.size()) ctx.binding_split_sizes.resize(ctx.pattern_N0s.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < ctx.pattern_N0s.bind_split_sizes.size(); ++bind_i) {
        ctx.binding_split_sizes[bind_i] = std::max<int64_t>(ctx.binding_split_sizes[bind_i], ctx.pattern_N0s.bind_split_sizes[bind_i]);
    }
    ctx.pattern_lambdas = dacpp::mpi::AccessPattern{};
    ctx.pattern_lambdas.param_id = 1;
    ctx.pattern_lambdas.name = "lambdas";
    ctx.pattern_lambdas.mode = dacpp::mpi::AccessMode::Read;
    ctx.pattern_lambdas.data_info.dim = lambdas.getDim();
    for (int dim = 0; dim < lambdas.getDim(); ++dim) ctx.pattern_lambdas.data_info.dimLength.push_back(lambdas.getShape(dim));
    Dac_Op pattern_lambdas_op_0;
    pattern_lambdas_op_0.setDimId(0);
    pattern_lambdas_op_0.size = 1;
    pattern_lambdas_op_0.stride = 1;
    pattern_lambdas_op_0.SetSplitSize(lambdas.getShape(0));
    ctx.pattern_lambdas.param_ops.push_back(pattern_lambdas_op_0);
    ctx.pattern_lambdas.bind_set_id.push_back(0);
    ctx.pattern_lambdas.bind_offset_expr.push_back("0");
    ctx.pattern_lambdas.is_index_op.push_back(true);
    ctx.pattern_lambdas.partition_shape = dacpp::mpi::init_partition_shape(ctx.pattern_lambdas);
    ctx.pattern_lambdas.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(ctx.pattern_lambdas);
    if (ctx.binding_split_sizes.size() < ctx.pattern_lambdas.bind_split_sizes.size()) ctx.binding_split_sizes.resize(ctx.pattern_lambdas.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < ctx.pattern_lambdas.bind_split_sizes.size(); ++bind_i) {
        ctx.binding_split_sizes[bind_i] = std::max<int64_t>(ctx.binding_split_sizes[bind_i], ctx.pattern_lambdas.bind_split_sizes[bind_i]);
    }
    ctx.pattern_local_A = dacpp::mpi::AccessPattern{};
    ctx.pattern_local_A.param_id = 2;
    ctx.pattern_local_A.name = "local_A";
    ctx.pattern_local_A.mode = dacpp::mpi::AccessMode::Write;
    ctx.pattern_local_A.data_info.dim = local_A.getDim();
    for (int dim = 0; dim < local_A.getDim(); ++dim) ctx.pattern_local_A.data_info.dimLength.push_back(local_A.getShape(dim));
    Dac_Op pattern_local_A_op_0;
    pattern_local_A_op_0.setDimId(0);
    pattern_local_A_op_0.size = 1;
    pattern_local_A_op_0.stride = 1;
    pattern_local_A_op_0.SetSplitSize(local_A.getShape(0));
    ctx.pattern_local_A.param_ops.push_back(pattern_local_A_op_0);
    ctx.pattern_local_A.bind_set_id.push_back(0);
    ctx.pattern_local_A.bind_offset_expr.push_back("0");
    ctx.pattern_local_A.is_index_op.push_back(true);
    ctx.pattern_local_A.partition_shape = dacpp::mpi::init_partition_shape(ctx.pattern_local_A);
    ctx.pattern_local_A.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(ctx.pattern_local_A);
    ctx.pattern_t = dacpp::mpi::AccessPattern{};
    ctx.pattern_t.param_id = 3;
    ctx.pattern_t.name = "t";
    ctx.pattern_t.mode = dacpp::mpi::AccessMode::Read;
    ctx.pattern_t.data_info.dim = t.getDim();
    for (int dim = 0; dim < t.getDim(); ++dim) ctx.pattern_t.data_info.dimLength.push_back(t.getShape(dim));
    ctx.pattern_t.partition_shape = dacpp::mpi::init_partition_shape(ctx.pattern_t);
    ctx.pattern_t.bind_split_sizes = dacpp::mpi::init_bind_split_sizes(ctx.pattern_t);
    if (ctx.binding_split_sizes.size() < ctx.pattern_t.bind_split_sizes.size()) ctx.binding_split_sizes.resize(ctx.pattern_t.bind_split_sizes.size(), 1);
    for (std::size_t bind_i = 0; bind_i < ctx.pattern_t.bind_split_sizes.size(); ++bind_i) {
        ctx.binding_split_sizes[bind_i] = std::max<int64_t>(ctx.binding_split_sizes[bind_i], ctx.pattern_t.bind_split_sizes[bind_i]);
    }
    if (ctx.binding_split_sizes.empty()) ctx.binding_split_sizes.push_back(1);
    for (int64_t split_size : ctx.binding_split_sizes) ctx.total_items *= split_size;
    ctx.item_range = dacpp::mpi::get_rank_item_range(ctx.total_items, ctx.mpi_rank, ctx.mpi_size);
    ctx.local_item_count = ctx.item_range.size();
    ctx.wave.use_span_pairs = false;
    ctx.wave.use_direct_kernel = false;
    ctx.wave.route_fast_paths_by_param.clear();
    ctx.wave.route_fast_paths_by_param.resize(4);
    ctx.wave.read_cache_transition_fast_paths.clear();
    ctx.wave.read_cache_transition_fast_paths.resize(0);
    ctx.wave.direct_kernel.slots.clear();
    ctx.wave.direct_kernel.slots_buffer.reset();
    ctx.wave.direct_kernel.next_stale_slots.clear();
    ctx.wave.direct_kernel.can_sparse_clear = false;
    ctx.pattern_N0s.bind_split_sizes = ctx.binding_split_sizes;
    ctx.plan_N0s = dacpp::mpi::build_input_pack_plan(ctx.item_range, ctx.pattern_N0s);
    ctx.local_N0s.resize(ctx.plan_N0s.pack.globals.size());
    dacpp::mpi::init_gathered_index_layout(ctx.input_layout_N0s, ctx.plan_N0s.pack.globals, ctx.mpi_rank, ctx.mpi_size);
    // DACPP fallback cached input: N0s
    if (ctx.mpi_rank == 0) {
        N0s.tensor2Array(ctx.global_N0s);
        dacpp::mpi::pack_values_by_globals_parallel_range_into(ctx.global_N0s, ctx.input_layout_N0s.globals.data(), ctx.input_layout_N0s.globals.size(), ctx.sendbuf_N0s);
    }
    MPI_Scatterv(ctx.mpi_rank == 0 ? ctx.sendbuf_N0s.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_N0s.counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_N0s.displs.data() : nullptr, MPI_DOUBLE, ctx.local_N0s.data(), ctx.input_layout_N0s.local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    ctx.pattern_lambdas.bind_split_sizes = ctx.binding_split_sizes;
    ctx.plan_lambdas = dacpp::mpi::build_input_pack_plan(ctx.item_range, ctx.pattern_lambdas);
    ctx.local_lambdas.resize(ctx.plan_lambdas.pack.globals.size());
    dacpp::mpi::init_gathered_index_layout(ctx.input_layout_lambdas, ctx.plan_lambdas.pack.globals, ctx.mpi_rank, ctx.mpi_size);
    // DACPP fallback cached input: lambdas
    if (ctx.mpi_rank == 0) {
        lambdas.tensor2Array(ctx.global_lambdas);
        dacpp::mpi::pack_values_by_globals_parallel_range_into(ctx.global_lambdas, ctx.input_layout_lambdas.globals.data(), ctx.input_layout_lambdas.globals.size(), ctx.sendbuf_lambdas);
    }
    MPI_Scatterv(ctx.mpi_rank == 0 ? ctx.sendbuf_lambdas.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_lambdas.counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.input_layout_lambdas.displs.data() : nullptr, MPI_DOUBLE, ctx.local_lambdas.data(), ctx.input_layout_lambdas.local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    ctx.pattern_local_A.bind_split_sizes = ctx.binding_split_sizes;
    ctx.plan_local_A = dacpp::mpi::build_output_pack_plan(ctx.item_range, ctx.pattern_local_A);
    ctx.local_local_A.resize(ctx.plan_local_A.pack.globals.size());
    const auto& writeback_globals_local_A = ctx.plan_local_A.pack.writeback_globals.empty() ? ctx.plan_local_A.pack.globals : ctx.plan_local_A.pack.writeback_globals;
    dacpp::mpi::init_gathered_index_layout(ctx.output_layout_local_A, writeback_globals_local_A, ctx.mpi_rank, ctx.mpi_size);
    dacpp::mpi::build_local_slots_for_globals(ctx.plan_local_A.pack, ctx.writeback_slots_local_A);
    ctx.writeback_values_local_A.resize(ctx.writeback_slots_local_A.size());
    if (ctx.mpi_rank == 0) ctx.global_recv_values_local_A.resize(ctx.output_layout_local_A.globals.size());
    ctx.pattern_t.bind_split_sizes = ctx.binding_split_sizes;
    ctx.plan_t = dacpp::mpi::build_input_pack_plan(ctx.item_range, ctx.pattern_t);
    ctx.local_t.resize(ctx.plan_t.pack.globals.size());
    dacpp::mpi::init_gathered_index_layout(ctx.input_layout_t, ctx.plan_t.pack.globals, ctx.mpi_rank, ctx.mpi_size);
    ctx.use_contiguous_kernel_views = true;
    ctx.use_contiguous_kernel_views = ctx.use_contiguous_kernel_views && dacpp::mpi::is_contiguous_kernel_pack_plan(ctx.plan_N0s, ctx.local_item_count, dacpp::mpi::partition_element_count(ctx.pattern_N0s));
    ctx.use_contiguous_kernel_views = ctx.use_contiguous_kernel_views && dacpp::mpi::is_contiguous_kernel_pack_plan(ctx.plan_lambdas, ctx.local_item_count, dacpp::mpi::partition_element_count(ctx.pattern_lambdas));
    ctx.use_contiguous_kernel_views = ctx.use_contiguous_kernel_views && dacpp::mpi::is_contiguous_kernel_pack_plan(ctx.plan_local_A, ctx.local_item_count, dacpp::mpi::partition_element_count(ctx.pattern_local_A));
    ctx.use_contiguous_kernel_views = ctx.use_contiguous_kernel_views && dacpp::mpi::is_contiguous_kernel_pack_plan(ctx.plan_t, ctx.local_item_count, dacpp::mpi::partition_element_count(ctx.pattern_t));
    ctx.use_partial_exchange = false;
    ctx.partial_exchange_disable_reason = "phase-c requires post-shell statements to lower as distributed followup or root-centric helpers";
}

void __dacpp_mpi_stencil_run___dacpp_mpi_stencil_DECAY_decay_0(__dacpp_mpi_stencil_ctx___dacpp_mpi_stencil_DECAY_decay_0& ctx, const dacpp::Vector<double> & N0s, const dacpp::Vector<double> & lambdas, dacpp::Vector<double> & local_A, const dacpp::Vector<double> & t) {
    int mpi_rank = ctx.mpi_rank;
    int mpi_size = ctx.mpi_size;
    dacpp::mpi::resetCollectPositionsProfile();
    auto dacpp_wrapper_start = std::chrono::steady_clock::now();
    auto& q = *ctx.q;
    const int64_t local_item_count = ctx.local_item_count;
    const bool dacpp_profile_enabled = dacpp::mpi::profilingEnabled();
    double dacpp_profile_input_ms = 0.0;
    double dacpp_profile_dist_setup_ms = 0.0;
    double dacpp_profile_kernel_ms = 0.0;
    double dacpp_profile_read_transition_ms = 0.0;
    double dacpp_profile_publish_ms = 0.0;
    double dacpp_profile_boundary_ms = 0.0;
    double dacpp_profile_root_bridge_ms = 0.0;
    double dacpp_profile_writeback_ms = 0.0;
    auto dacpp_profile_now = [&]() {
        return dacpp_profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    };
    auto dacpp_profile_add = [&](double& bucket, std::chrono::steady_clock::time_point start) {
        if (dacpp_profile_enabled) {
            bucket += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
    };
    if (!ctx.use_partial_exchange) {
    auto dacpp_profile_input_start = dacpp_profile_now();
    auto& local_N0s = ctx.local_N0s;
    // DACPP fallback cached input reused: N0s
    const int N0s_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(ctx.pattern_N0s));
    auto& local_lambdas = ctx.local_lambdas;
    // DACPP fallback cached input reused: lambdas
    const int lambdas_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(ctx.pattern_lambdas));
    auto& local_local_A = ctx.local_local_A;
    local_local_A.assign(ctx.plan_local_A.pack.globals.size(), double{});
    const int local_A_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(ctx.pattern_local_A));
    auto& local_t = ctx.local_t;
    auto& input_layout_t = ctx.input_layout_t;
    auto& global_t = ctx.global_t;
    auto& sendbuf_t = ctx.sendbuf_t;
    if (mpi_rank == 0) {
        t.tensor2Array(global_t);
        dacpp::mpi::pack_values_by_globals_parallel_range_into(global_t, input_layout_t.globals.data(), input_layout_t.globals.size(), sendbuf_t);
    }
    local_t.resize(static_cast<std::size_t>(input_layout_t.local_count));
    MPI_Scatterv(mpi_rank == 0 ? sendbuf_t.data() : nullptr, mpi_rank == 0 ? input_layout_t.counts.data() : nullptr, mpi_rank == 0 ? input_layout_t.displs.data() : nullptr, MPI_DOUBLE, local_t.data(), input_layout_t.local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const int t_partition_size = static_cast<int>(dacpp::mpi::partition_element_count(ctx.pattern_t));
    dacpp_profile_add(dacpp_profile_input_ms, dacpp_profile_input_start);
    auto dacpp_profile_kernel_start = dacpp_profile_now();
    if (local_item_count > 0) {
        {
            sycl::buffer<double, 1> buffer_N0s(local_N0s.data(), sycl::range<1>(local_N0s.size()));
            sycl::buffer<int32_t, 1> slots_buffer_N0s(ctx.plan_N0s.compact_slots.data(), sycl::range<1>(ctx.plan_N0s.compact_slots.size()));
            sycl::buffer<int32_t, 1> key_offsets_buffer_N0s(ctx.plan_N0s.item_key_offsets.data(), sycl::range<1>(ctx.plan_N0s.item_key_offsets.size()));
            sycl::buffer<double, 1> buffer_lambdas(local_lambdas.data(), sycl::range<1>(local_lambdas.size()));
            sycl::buffer<int32_t, 1> slots_buffer_lambdas(ctx.plan_lambdas.compact_slots.data(), sycl::range<1>(ctx.plan_lambdas.compact_slots.size()));
            sycl::buffer<int32_t, 1> key_offsets_buffer_lambdas(ctx.plan_lambdas.item_key_offsets.data(), sycl::range<1>(ctx.plan_lambdas.item_key_offsets.size()));
            sycl::buffer<double, 1> buffer_local_A(local_local_A.data(), sycl::range<1>(local_local_A.size()));
            sycl::buffer<int32_t, 1> slots_buffer_local_A(ctx.plan_local_A.compact_slots.data(), sycl::range<1>(ctx.plan_local_A.compact_slots.size()));
            sycl::buffer<int32_t, 1> key_offsets_buffer_local_A(ctx.plan_local_A.item_key_offsets.data(), sycl::range<1>(ctx.plan_local_A.item_key_offsets.size()));
            sycl::buffer<double, 1> buffer_t(local_t.data(), sycl::range<1>(local_t.size()));
            sycl::buffer<int32_t, 1> slots_buffer_t(ctx.plan_t.compact_slots.data(), sycl::range<1>(ctx.plan_t.compact_slots.size()));
            sycl::buffer<int32_t, 1> key_offsets_buffer_t(ctx.plan_t.item_key_offsets.data(), sycl::range<1>(ctx.plan_t.item_key_offsets.size()));
            q.submit([&](sycl::handler& h) {
                auto acc_N0s = buffer_N0s.get_access<sycl::access::mode::read>(h);
                auto slots_acc_N0s = slots_buffer_N0s.get_access<sycl::access::mode::read>(h);
                auto key_offsets_acc_N0s = key_offsets_buffer_N0s.get_access<sycl::access::mode::read>(h);
                auto acc_lambdas = buffer_lambdas.get_access<sycl::access::mode::read>(h);
                auto slots_acc_lambdas = slots_buffer_lambdas.get_access<sycl::access::mode::read>(h);
                auto key_offsets_acc_lambdas = key_offsets_buffer_lambdas.get_access<sycl::access::mode::read>(h);
                auto acc_local_A = buffer_local_A.get_access<sycl::access::mode::read_write>(h);
                auto slots_acc_local_A = slots_buffer_local_A.get_access<sycl::access::mode::read>(h);
                auto key_offsets_acc_local_A = key_offsets_buffer_local_A.get_access<sycl::access::mode::read>(h);
                auto acc_t = buffer_t.get_access<sycl::access::mode::read>(h);
                auto slots_acc_t = slots_buffer_t.get_access<sycl::access::mode::read>(h);
                auto key_offsets_acc_t = key_offsets_buffer_t.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(static_cast<std::size_t>(local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* data_N0s = acc_N0s.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_N0s = slots_acc_N0s.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* key_offsets_N0s = key_offsets_acc_N0s.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const double> view_N0s{data_N0s, slots_N0s, key_offsets_N0s[item_linear]};
                    auto* data_lambdas = acc_lambdas.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_lambdas = slots_acc_lambdas.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* key_offsets_lambdas = key_offsets_acc_lambdas.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const double> view_lambdas{data_lambdas, slots_lambdas, key_offsets_lambdas[item_linear]};
                    auto* data_local_A = acc_local_A.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_local_A = slots_acc_local_A.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* key_offsets_local_A = key_offsets_acc_local_A.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<double> view_local_A{data_local_A, slots_local_A, key_offsets_local_A[item_linear]};
                    auto* data_t = acc_t.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* slots_t = slots_acc_t.template get_multi_ptr<sycl::access::decorated::no>().get();
                    auto* key_offsets_t = key_offsets_acc_t.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::View1D<const double> view_t{data_t, slots_t, key_offsets_t[item_linear]};
                    decay_mpi_local(view_N0s, view_lambdas, view_local_A, view_t);
                });
            });
            q.wait();
        }
    }
    dacpp_profile_add(dacpp_profile_kernel_ms, dacpp_profile_kernel_start);
    auto dacpp_profile_writeback_start = dacpp_profile_now();
    auto& output_layout_local_A = ctx.output_layout_local_A;
    auto& writeback_values_local_A = ctx.writeback_values_local_A;
    auto& global_recv_values_local_A = ctx.global_recv_values_local_A;
    auto& global_out_local_A = ctx.global_out_local_A;
    dacpp::mpi::pack_values_by_slots_parallel_into(local_local_A, ctx.writeback_slots_local_A, writeback_values_local_A);
    MPI_Gatherv(writeback_values_local_A.data(), output_layout_local_A.local_count, MPI_DOUBLE, mpi_rank == 0 ? global_recv_values_local_A.data() : nullptr, mpi_rank == 0 ? output_layout_local_A.counts.data() : nullptr, mpi_rank == 0 ? output_layout_local_A.displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        local_A.tensor2Array(global_out_local_A);
        dacpp::mpi::apply_writeback_by_globals(global_recv_values_local_A, output_layout_local_A.globals, global_out_local_A);
        local_A.array2Tensor(global_out_local_A);
    } else {
    }
    dacpp_profile_add(dacpp_profile_writeback_ms, dacpp_profile_writeback_start);
    }
    if (dacpp::mpi::profilingEnabled()) {
        auto dacpp_wrapper_end = std::chrono::steady_clock::now();
        double dacpp_wrapper_local_ms = std::chrono::duration<double, std::milli>(dacpp_wrapper_end - dacpp_wrapper_start).count();
        double dacpp_wrapper_max_ms = 0.0;
        MPI_Reduce(&dacpp_wrapper_local_ms, &dacpp_wrapper_max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        double dacpp_profile_local_parts[8] = {dacpp_profile_input_ms, dacpp_profile_dist_setup_ms, dacpp_profile_kernel_ms, dacpp_profile_read_transition_ms, dacpp_profile_publish_ms, dacpp_profile_boundary_ms, dacpp_profile_root_bridge_ms, dacpp_profile_writeback_ms};
        double dacpp_profile_max_parts[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        MPI_Reduce(dacpp_profile_local_parts, dacpp_profile_max_parts, 8, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (mpi_rank == 0) {
        std::fprintf(stderr, "[DACPP][PROFILE][%s] wrapper_total_ms(max): %.3f\n", "__dacpp_mpi_stencil_DECAY_decay_0", dacpp_wrapper_max_ms);
        std::fprintf(stderr, "[DACPP][PROFILE][%s] run_breakdown_ms(max): input=%.3f dist_setup=%.3f kernel=%.3f read_transition=%.3f publish=%.3f boundary=%.3f root_bridge=%.3f writeback=%.3f\n", "__dacpp_mpi_stencil_DECAY_decay_0", dacpp_profile_max_parts[0], dacpp_profile_max_parts[1], dacpp_profile_max_parts[2], dacpp_profile_max_parts[3], dacpp_profile_max_parts[4], dacpp_profile_max_parts[5], dacpp_profile_max_parts[6], dacpp_profile_max_parts[7]);
        }
    }
    dacpp::mpi::reportCollectPositionsProfile("__dacpp_mpi_stencil_DECAY_decay_0", MPI_COMM_WORLD);
}

void __dacpp_mpi_stencil_materialize___dacpp_mpi_stencil_DECAY_decay_0(__dacpp_mpi_stencil_ctx___dacpp_mpi_stencil_DECAY_decay_0& ctx, const dacpp::Vector<double> & N0s, const dacpp::Vector<double> & lambdas, dacpp::Vector<double> & local_A, const dacpp::Vector<double> & t) {
    if (!ctx.use_partial_exchange) {
        return;
    }
    return;
}

void __dacpp_mpi_stencil_DECAY_decay_0(const dacpp::Vector<double> & N0s, const dacpp::Vector<double> & lambdas, dacpp::Vector<double> & local_A, const dacpp::Vector<double> & t) {
    __dacpp_mpi_stencil_ctx___dacpp_mpi_stencil_DECAY_decay_0 ctx;
    __dacpp_mpi_stencil_init___dacpp_mpi_stencil_DECAY_decay_0(ctx, N0s, lambdas, local_A, t);
    __dacpp_mpi_stencil_run___dacpp_mpi_stencil_DECAY_decay_0(ctx, N0s, lambdas, local_A, t);
    __dacpp_mpi_stencil_materialize___dacpp_mpi_stencil_DECAY_decay_0(ctx, N0s, lambdas, local_A, t);
}

using __dacpp_mpi_stencil_ctx_DECAY_decay = __dacpp_mpi_stencil_ctx___dacpp_mpi_stencil_DECAY_decay_0;
void __dacpp_mpi_stencil_init_DECAY_decay(__dacpp_mpi_stencil_ctx_DECAY_decay& ctx, const dacpp::Vector<double> & N0s, const dacpp::Vector<double> & lambdas, dacpp::Vector<double> & local_A, const dacpp::Vector<double> & t) {
    __dacpp_mpi_stencil_init___dacpp_mpi_stencil_DECAY_decay_0(ctx, N0s, lambdas, local_A, t);
}
void __dacpp_mpi_stencil_run_DECAY_decay(__dacpp_mpi_stencil_ctx_DECAY_decay& ctx, const dacpp::Vector<double> & N0s, const dacpp::Vector<double> & lambdas, dacpp::Vector<double> & local_A, const dacpp::Vector<double> & t) {
    __dacpp_mpi_stencil_run___dacpp_mpi_stencil_DECAY_decay_0(ctx, N0s, lambdas, local_A, t);
}
void __dacpp_mpi_stencil_materialize_DECAY_decay(__dacpp_mpi_stencil_ctx_DECAY_decay& ctx, const dacpp::Vector<double> & N0s, const dacpp::Vector<double> & lambdas, dacpp::Vector<double> & local_A, const dacpp::Vector<double> & t) {
    __dacpp_mpi_stencil_materialize___dacpp_mpi_stencil_DECAY_decay_0(ctx, N0s, lambdas, local_A, t);
}


void calculateDecay(const std::vector<double>& lambdas, const std::vector<double>& N0s,
                    double dt, double T) {
    size_t numIsotopes = lambdas.size();
    std::vector<double> A(static_cast<size_t>(T / dt) * numIsotopes, 0.0);
    std::vector<double> t;
    t.push_back(0.0);
    std::vector<double> local_A(numIsotopes, 0.0);
    dacpp::Vector<double> local_A_tensor(local_A);
    dacpp::Vector<double> N0s_tensor(N0s);
    dacpp::Vector<double> lambdas_tensor(lambdas);
    dacpp::Vector<double> t_tensor(t);
    dacpp::Matrix<double> A_tensor({static_cast<int>(T / dt), static_cast<int>(numIsotopes)}, A);

        __dacpp_mpi_stencil_ctx___dacpp_mpi_stencil_DECAY_decay_0 __dacpp_mpi_stencil_ctx_0;
    __dacpp_mpi_stencil_init___dacpp_mpi_stencil_DECAY_decay_0(__dacpp_mpi_stencil_ctx_0, N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);
while (t_tensor[0] <= T) {
        __dacpp_mpi_stencil_run___dacpp_mpi_stencil_DECAY_decay_0(__dacpp_mpi_stencil_ctx_0, N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);
        A_tensor[10 * t_tensor[0]] = local_A_tensor;
        t_tensor[0] += dt;
    }
    __dacpp_mpi_stencil_materialize___dacpp_mpi_stencil_DECAY_decay_0(__dacpp_mpi_stencil_ctx_0, N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);

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

    std::vector<double> lambdas(numIsotopes);
    std::vector<double> N0s(numIsotopes, 1000.0);
    for (size_t i = 0; i < numIsotopes; ++i) lambdas[i] = 0.01 + 0.01 * i;

    double __t0 = MPI_Wtime();
    calculateDecay(lambdas, N0s, dt, T);
    double __el = MPI_Wtime() - __t0;
    __dacpp_e2e("decay.ablation", __el);
    
    if (dacpp_mpi_finalize_needed) {
        MPI_Finalize();
        dacpp_mpi_finalize_needed = 0;
    }
return 0;
}

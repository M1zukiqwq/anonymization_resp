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


struct __dacpp_mpi_or_DECAY_decay_0_ctx {
    int mpi_rank = 0;
    int mpi_size = 1;
    int64_t __or_total_items = 0;
    int64_t __or_local_item_count = 0;
    dacpp::mpi::operator_resident::RankRange1D __or_range{};
    std::vector<int> __or_counts;
    std::vector<int> __or_displs;
    dacpp::mpi::SegmentedProfile __or_profile;
    sycl::queue& q = dacpp::mpi::operator_resident::default_queue();
    std::vector<double> __or_local_N0s;
    std::vector<double> __or_local_lambdas;
    std::vector<double> __or_local_local_A;
    double __or_scalar_t{};
    std::vector<double> __or_local_t;
};
void __dacpp_mpi_or_DECAY_decay_0_init(__dacpp_mpi_or_DECAY_decay_0_ctx& ctx, const dacpp::Vector<double> & __or_arg0, const dacpp::Vector<double> & __or_arg1, dacpp::Vector<double> & __or_arg2, const dacpp::Vector<double> & __or_arg3) {
    auto dacpp_profile_init_start = dacpp::mpi::profileSegmentStart();
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);
    ctx.__or_total_items = __or_arg2.getShape(0);
    ctx.__or_range = dacpp::mpi::operator_resident::rank_range_1d(ctx.__or_total_items, ctx.mpi_rank, ctx.mpi_size);
    ctx.__or_local_item_count = ctx.__or_range.count;
    dacpp::mpi::operator_resident::counts_displs_1d(ctx.__or_total_items, ctx.mpi_size, ctx.__or_counts, ctx.__or_displs);
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Init, dacpp_profile_init_start);
    auto dacpp_profile_scatter_start_N0s = dacpp::mpi::profileSegmentStart();
    ctx.__or_local_N0s.resize(static_cast<std::size_t>(ctx.__or_local_item_count));
    std::vector<double> __or_global_N0s;
    if (ctx.mpi_rank == 0) {
        __or_arg0.tensor2Array(__or_global_N0s);
    }
    MPI_Scatterv(ctx.mpi_rank == 0 ? __or_global_N0s.data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, MPI_DOUBLE, ctx.__or_local_N0s.data(), static_cast<int>(ctx.__or_local_item_count), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Scatter, dacpp_profile_scatter_start_N0s);
    auto dacpp_profile_scatter_start_lambdas = dacpp::mpi::profileSegmentStart();
    ctx.__or_local_lambdas.resize(static_cast<std::size_t>(ctx.__or_local_item_count));
    std::vector<double> __or_global_lambdas;
    if (ctx.mpi_rank == 0) {
        __or_arg1.tensor2Array(__or_global_lambdas);
    }
    MPI_Scatterv(ctx.mpi_rank == 0 ? __or_global_lambdas.data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, MPI_DOUBLE, ctx.__or_local_lambdas.data(), static_cast<int>(ctx.__or_local_item_count), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Scatter, dacpp_profile_scatter_start_lambdas);
    ctx.__or_local_local_A.assign(static_cast<std::size_t>(ctx.__or_local_item_count), double{});
}
void __dacpp_mpi_or_DECAY_decay_0_run(__dacpp_mpi_or_DECAY_decay_0_ctx& ctx, const dacpp::Vector<double> & __or_arg0, const dacpp::Vector<double> & __or_arg1, dacpp::Vector<double> & __or_arg2, const dacpp::Vector<double> & __or_arg3) {
    if (__or_arg3.getSize() != 1) {
        if (ctx.mpi_rank == 0) std::fprintf(stderr, "[DACPP][MPI][OR][P4.5] scalar parameter t expected size 1\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    ctx.__or_scalar_t = double{};
    // P4.5 loop-lowered direct scalar reader: all ranks execute the host scalar update, so refresh from the local replicated tensor without per-iteration MPI_Bcast.
    std::vector<double> __or_scalar_vec_t;
    __or_arg3.tensor2Array(__or_scalar_vec_t);
    if (!__or_scalar_vec_t.empty()) ctx.__or_scalar_t = __or_scalar_vec_t[0];
    ctx.__or_local_t.assign(1, ctx.__or_scalar_t);
    auto dacpp_profile_kernel_start = dacpp::mpi::profileSegmentStart();
    const int64_t __or_local_item_count = ctx.__or_local_item_count;
    auto& q = ctx.q;
    if (__or_local_item_count > 0) {
        {
            sycl::buffer<double, 1> __or_buffer_N0s(ctx.__or_local_N0s.data(), sycl::range<1>(ctx.__or_local_N0s.size()));
            sycl::buffer<double, 1> __or_buffer_lambdas(ctx.__or_local_lambdas.data(), sycl::range<1>(ctx.__or_local_lambdas.size()));
            sycl::buffer<double, 1> __or_buffer_local_A(ctx.__or_local_local_A.data(), sycl::range<1>(ctx.__or_local_local_A.size()));
            sycl::buffer<double, 1> __or_buffer_t(ctx.__or_local_t.data(), sycl::range<1>(ctx.__or_local_t.size()));
            q.submit([&](sycl::handler& h) {
                auto __or_acc_N0s = __or_buffer_N0s.get_access<sycl::access::mode::read>(h);
                auto __or_acc_lambdas = __or_buffer_lambdas.get_access<sycl::access::mode::read>(h);
                auto __or_acc_local_A = __or_buffer_local_A.get_access<sycl::access::mode::read_write>(h);
                auto __or_acc_t = __or_buffer_t.get_access<sycl::access::mode::read>(h);
                h.parallel_for<class __dacpp_k0>(sycl::range<1>(static_cast<std::size_t>(__or_local_item_count)), [=](sycl::id<1> idx) {
                    const int item_linear = static_cast<int>(idx[0]);
                    auto* __or_data_N0s = __or_acc_N0s.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<const double> view_N0s{__or_data_N0s, item_linear};
                    auto* __or_data_lambdas = __or_acc_lambdas.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<const double> view_lambdas{__or_data_lambdas, item_linear};
                    auto* __or_data_local_A = __or_acc_local_A.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<double> view_local_A{__or_data_local_A, item_linear};
                    auto* __or_data_t = __or_acc_t.template get_multi_ptr<sycl::access::decorated::no>().get();
                    dacpp::mpi::ContiguousView1D<const double> view_t{__or_data_t, 0};
                    decay_mpi_local(view_N0s, view_lambdas, view_local_A, view_t);
                });
            });
            q.wait();
        }
    }
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Kernel, dacpp_profile_kernel_start);
    // No downstream resident reader for local_A; host materialization below preserves visibility.
    std::vector<double> __or_materialized_local_A;
    if (ctx.mpi_rank == 0) {
        __or_materialized_local_A.resize(static_cast<std::size_t>(ctx.__or_total_items));
    }
    auto dacpp_profile_gather_start_local_A = dacpp::mpi::profileSegmentStart();
    MPI_Gatherv(ctx.__or_local_local_A.data(), static_cast<int>(ctx.__or_local_item_count), MPI_DOUBLE, ctx.mpi_rank == 0 ? __or_materialized_local_A.data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_counts.data() : nullptr, ctx.mpi_rank == 0 ? ctx.__or_displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Gather, dacpp_profile_gather_start_local_A);
    auto dacpp_profile_materialize_start_local_A = dacpp::mpi::profileSegmentStart();
    if (ctx.mpi_rank == 0) {
        __or_arg2.array2Tensor(__or_materialized_local_A);
    }
    dacpp::mpi::recordProfileSegment(ctx.__or_profile, dacpp::mpi::ProfileSegment::Materialize, dacpp_profile_materialize_start_local_A);
}
void __dacpp_mpi_or_DECAY_decay_0_materialize(__dacpp_mpi_or_DECAY_decay_0_ctx& ctx, const dacpp::Vector<double> & __or_arg0, const dacpp::Vector<double> & __or_arg1, dacpp::Vector<double> & __or_arg2, const dacpp::Vector<double> & __or_arg3) {
    (void)ctx;
    dacpp::mpi::reportSegmentedProfile("__dacpp_mpi_or_DECAY_decay_0_materialize", ctx.__or_profile, MPI_COMM_WORLD);
}
void __dacpp_mpi_or_DECAY_decay_0(const dacpp::Vector<double> & __or_arg0, const dacpp::Vector<double> & __or_arg1, dacpp::Vector<double> & __or_arg2, const dacpp::Vector<double> & __or_arg3) {
    __dacpp_mpi_or_DECAY_decay_0_ctx ctx;
    __dacpp_mpi_or_DECAY_decay_0_init(ctx, __or_arg0, __or_arg1, __or_arg2, __or_arg3);
    __dacpp_mpi_or_DECAY_decay_0_run(ctx, __or_arg0, __or_arg1, __or_arg2, __or_arg3);
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

        __dacpp_mpi_or_DECAY_decay_0_ctx __dacpp_mpi_or_ctx_0;
    __dacpp_mpi_or_DECAY_decay_0_init(__dacpp_mpi_or_ctx_0, N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);
while (t_tensor[0] <= T) {
        __dacpp_mpi_or_DECAY_decay_0_run(__dacpp_mpi_or_ctx_0, N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);
        A_tensor[10 * t_tensor[0]] = local_A_tensor;
        t_tensor[0] += dt;
    }
    __dacpp_mpi_or_DECAY_decay_0_materialize(__dacpp_mpi_or_ctx_0, N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor);

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

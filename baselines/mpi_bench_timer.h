// Self-contained MPI timing harness for Tier-1 hand-written baselines.
// Output format matches the DACPP _test harness so results are directly
// comparable:  [MPI TEST] <name> | ranks=N | total_max=.. | total_avg=.. |
//              communication_max=.. | communication_avg=.. |
//              computation_max=.. | computation_avg=..
#ifndef MPI_BENCH_TIMER_H
#define MPI_BENCH_TIMER_H
#include <mpi.h>
#include <cstdio>

struct BenchTimer {
    const char* name;
    int rank = 0, size = 1;
    double t0 = 0.0, comm = 0.0, comp = 0.0, seg_ = 0.0;
    explicit BenchTimer(const char* n) : name(n) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        t0 = MPI_Wtime();
    }
    void comm_begin() { seg_ = MPI_Wtime(); }
    void comm_end()   { comm += MPI_Wtime() - seg_; }
    void comp_begin() { seg_ = MPI_Wtime(); }
    void comp_end()   { comp += MPI_Wtime() - seg_; }
    void report() {
        const double tot = MPI_Wtime() - t0;
        double tmax = 0, tsum = 0, cmax = 0, csum = 0, kmax = 0, ksum = 0;
        MPI_Reduce(&tot,  &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&tot,  &tsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comm, &cmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comm, &csum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comp, &kmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comp, &ksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::printf("[MPI TEST] %s | ranks=%d | total_max=%.6f s | "
                        "total_avg=%.6f s | communication_max=%.6f s | "
                        "communication_avg=%.6f s | computation_max=%.6f s | "
                        "computation_avg=%.6f s\n",
                        name, size, tmax, tsum / size, cmax, csum / size,
                        kmax, ksum / size);
        }
    }
};
#endif

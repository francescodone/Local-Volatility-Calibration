#ifndef OPENMP_UTILITIES
#define OPENMP_UTILITIES

//#include <stdio.h>
#include <omp.h>
#include <assert.h>

int get_CPU_num_threads() {
    int procs;

#pragma omp parallel shared(procs)
    {
        int th_id = omp_get_thread_num();
        if(th_id == 0) { procs = omp_get_num_threads(); }
    }

    bool valid_procs = (procs > 0) && (procs <= 1024);
    assert(valid_procs && "Number of threads NOT in {1, ..., 1024}");
    return procs;
}

#endif //OPENMP_UTILITIES

#include <x86intrin.h> // for rdtsc
#include <stdint.h> // for uint64_t
extern uint64_t g_t_prefetch;
extern uint64_t g_n_prefetch;
#include "matmul_prefetch.h"

void 
matmul_prefetch(
    float * restrict xout, 
    const float * const x,
    const float * const w,
    int n, 
    int d
    ) 
{
  uint64_t t = __rdtsc();
#pragma omp parallel for schedule(static, 8)
  for ( register int i  = 0; i < d; i++) {
    register const float * const w_i = w + (i*n);
    for ( register int off = 0; off < n; off += 64 ) { 
    __builtin_prefetch(w_i + n, 0, 0); 
    }
  }
  g_t_prefetch += (__rdtsc() - t);
  g_n_prefetch ++;
}

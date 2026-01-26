#include <x86intrin.h> // for rdtsc
#include <stdint.h> // for uint64_t
extern uint64_t g_t_matmul; // for timing 
extern uint64_t g_n_matmul; // for timing 

#include "dot_prod.h"
#include "dot_prod_256.h"
#include "matmul_ispc_wrap.h"
void 
matmul(
    float * restrict xout, 
    const float * const x,
    const float * const w,
    int n, 
    int d
    ) 
{
  uint64_t t = __rdtsc();
  // W (d,n) @ x (n,) -> xout (d,)
  // TODO Is there a value in setting chunk_size to 8
  // Preliminary experimentation suggests minor gain in doing so
#pragma omp parallel for schedule(static, 8)
  for ( register int i  = 0; i < d; i++) {
    register const float * const w_i = w + (i*n);
    // __builtin_prefetch(w_i + n, 0); 
    dot_prod(x, w_i, n, &(xout[i]));
    // No major improvement seen with following:
    // xout[i] = dot_product_fma_avx2(x, w_i, n);
  }
  g_t_matmul += (__rdtsc() - t);
  g_n_matmul += (uint64_t)(d * n * 2); 
}

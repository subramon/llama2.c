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
#pragma omp parallel for 
  for (  int i  = 0; i < d; i++) {
    const float * const w_i = w + (i*n);
    // __builtin_prefetch(w_i + n, 0); slows things down, hence commented
    dot_prod(x, w_i, n, &(xout[i]));
    // Strangely, following was waaaay slower.
    // xout[i] = dot_product_fma_avx2(x, w_i, n);
  }
  g_t_matmul += (__rdtsc() - t);
  g_n_matmul += d * n * 2; 
}

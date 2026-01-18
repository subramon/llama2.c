#include <x86intrin.h> // for rdtsc
#include <stdint.h> // for uint64_t
extern uint64_t g_t_matmul; // for timing 
extern uint64_t g_n_matmul; // for timing 

#include "dot_prod_qnt.h"
#include "matmul_qnt_ispc_wrap.h"
void 
matmul(
    float * restrict xout, 
    const float * const x,
    const uint8_t * const w,
    const float * const offset,
    const float * const delta,
    int n, 
    int d
    ) 
{
  uint64_t t = __rdtsc();
  // W (d,n) @ x (n,) -> xout (d,)
#pragma omp parallel for 
  for (  int i  = 0; i < d; i++) {
    const uint8_t * const w_i = w + (i*n);
    float offset_i = offset[i];
    float delta_i  = delta[i];
    // __builtin_prefetch(w_i + n, 0); slows things down, hence commented
    dot_prod_qnt(offset_i, delta_i, x, w_i, n, &(xout[i]));
  }
  g_t_matmul += (__rdtsc() - t);
  g_n_matmul += (uint64_t)(d * n * 2); 
}

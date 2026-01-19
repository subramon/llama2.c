#include <x86intrin.h> // for rdtsc
#include <stdint.h> // for uint64_t
extern uint64_t g_t_matmul; // for timing 
extern uint64_t g_n_matmul; // for timing 
#include "matmul_qnt.h"
void 
matmul_qnt(
    float * restrict xout, 
    const float * const x, 
    const float * const wf32, 
    const uint8_t * const wui8, 
    const float * const offset, 
    const float * const delta, 
    int n, 
    int d
    ) 
{
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int i;
  uint64_t t = __rdtsc();
  for (i = 0; i < d; i++) {
    register float val = 0.0f;
    const uint8_t *w_i = wui8 + (i*n);
    const float *wf32_i = wf32 + (i*n);
    float offset_i = offset[i];
    float delta_i  = delta[i];
    for (int j = 0; j < n; j++) {
      float w_ij = offset_i + ( w_i[j] * delta_i );
#ifdef DEBUG
      // TODO Check difference between w_ij and wf32_i[j]
#endif
      val += w_ij * x[j];
    }
    xout[i] = val;
  }
  g_t_matmul += (__rdtsc() - t);
  g_n_matmul += (n*d);
}


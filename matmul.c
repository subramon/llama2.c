#include <x86intrin.h> // for rdtsc
#include <stdint.h> // for uint64_t
extern uint64_t g_t_matmul; // for timing 
extern uint64_t g_n_matmul; // for timing 
#include "matmul.h"
void 
matmul(
    float * restrict xout, 
    const float * const x, 
    const float * const w, 
    int n, 
    int d
    ) 
{
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  uint64_t t = __rdtsc();
#pragma omp parallel for 
  for ( int i = 0; i < d; i++) {
    register float val = 0.0f;
    register float *w_i = w + (i*n);
    for (int j = 0; j < n; j++) {
      val += w_i[j] * x[j];
    }
    xout[i] = val;
  }
  g_t_matmul += (__rdtsc() - t);
  g_n_matmul += (2*n*d);
}


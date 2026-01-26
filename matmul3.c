#include <x86intrin.h> // for rdtsc
#include <stdint.h> // for uint64_t
extern uint64_t g_t_matmul; // for timing 
extern uint64_t g_n_matmul; // for timing 
#include "matmul3.h"
void 
matmul3(
    float * restrict xout1, 
    float * restrict xout2, 
    float * restrict xout3, 
    const float * const x1, 
    const float * const x2, 
    const float * const x3, 
    const float * const w1, 
    const float * const w2, 
    const float * const w3, 
    int n, 
    int d
    ) 
{
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  uint64_t t = __rdtsc();
#pragma omp parallel for 
  for ( int i = 0; i < 3*d; i++) {
    register float *w_i  = NULL;
    register float *x    = NULL;
    register float *xout = NULL;
    register int iprime;
    if ( i < d ) { 
      iprime = i; w_i = w1 + (iprime*n); x = x1; xout = xout1; 
    }
    else if ( i < 2*d ) { 
      iprime = i - d; w_i = w2 + (iprime*n); x = x2; xout = xout2; 
    }
    else { 
      iprime = i - 2*d; w_i = w3 + (iprime*n); x = x3; xout = xout3; 
    }
    register float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w_i[j] * x[j];
    }
    xout[iprime] = val;
  }
  g_t_matmul += (__rdtsc() - t);
  g_n_matmul += (3*2*n*d);
}


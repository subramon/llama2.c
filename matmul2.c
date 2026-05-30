#include <stdint.h> // for uint64_t
#include <stdlib.h> // for NULL
#include "matmul2.h"
void 
matmul2(
    float * restrict xout1, 
    float * restrict xout2, 
    const float * const x1, 
    const float * const x2, 
    const float * const w1, 
    const float * const w2, 
    int n, 
    int d
    ) 
{
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
#pragma omp parallel for 
  for ( int i = 0; i < 2*d; i++) {
    float *w_i  = NULL;
    float *x    = NULL;
    float *xout = NULL;
    int iprime;
    if ( i < d ) { 
      iprime = i;     w_i = w1 + (iprime*n); x = x1; xout = xout1;
    }
    else {
      iprime = i - d; w_i = w2 + (iprime*n); x = x2; xout = xout2;
    }
    register float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w_i[j] * x[j];
    }
    xout[iprime] = val;
  }
}


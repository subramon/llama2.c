#include "dot_prod_ispc.h"
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
  // W (d,n) @ x (n,) -> xout (d,)
#pragma omp parallel for 
  for (  int i  = 0; i < d; i++) {
    const float * const w_i = w + (i*n);
    dot_prod(x, w_i, n, &(xout[i]));
  }
}

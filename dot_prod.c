#include "dot_prod.h"
void 
dot_prod(
    const float * const x, 
    const float * const y, 
    int n,
    float * restrict ptr_rslt
    ) 
{
  register float sum = 0.0f;
  for ( int i = 0; i < n; i++ ) { 
    sum += x[i] * y[i];
  }
  *ptr_rslt = sum;
}

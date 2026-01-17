#include "consts.h"
export void 
add_v(
    __attribute__((aligned(BYTES_IN_REG))) uniform float x[],
    __attribute__((aligned(BYTES_IN_REG))) uniform float y[], 
    uniform int n
    ) 
{
  assume(n % programCount == 0);
  foreach ( i = 0 ... n ) {
    // prefetch_l1(&y[i+32]);
    // prefetchw_l1(&x[i+32]);
    x[i] += y[i];
  }
}

#include "consts.h"
__attribute__((external_only))
export void 
add_v(
    __attribute__((aligned(BYTES_IN_REG))) uniform float x[],
    __attribute__((aligned(BYTES_IN_REG))) uniform float y[], 
    uniform int n
    ) 
{
  assume(((uniform uint64)((void*)x) & (32 * TARGET_WIDTH)-1) == 0);
  assume(((uniform uint64)((void*)y) & (32 * TARGET_WIDTH)-1) == 0);
  assume(n % programCount == 0);
  foreach ( i = 0 ... n ) {
    // prefetch_l1(&y[i+32]);
    // prefetchw_l1(&x[i+32]);
    x[i] += y[i];
  }
}

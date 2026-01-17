#include "consts.h"
// x_i += scalar * y_i
__attribute__((external_only))
export void 
mul_v_add_s(
    __attribute__((aligned(BYTES_IN_REG))) uniform float x[], 
    uniform float scalar,
    __attribute__((aligned(BYTES_IN_REG))) uniform float y[], 
    uniform int n
    ) 
{
  assume(n % programCount == 0);
  assume(((uniform uint64)((void*)x) & (32 * TARGET_WIDTH)-1) == 0);
  assume(((uniform uint64)((void*)y) & (32 * TARGET_WIDTH)-1) == 0);
  foreach ( i = 0 ... n ) { 
    // prefetch_l1(&(y[i+32]));
    // prefetchw_l1(&(x[i+32]));
    x[i] += scalar * y[i];
  }
}

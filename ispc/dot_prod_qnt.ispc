#include "consts.h"

__attribute__((external_only))
export void 
dot_prod_qnt(
    uniform float offset,
    uniform float delta,
    __attribute__((aligned(BYTES_IN_REG))) uniform float x[], 
    __attribute__((aligned(BYTES_IN_REG))) uniform uint8 y[], 
    uniform int n,
    __attribute__((aligned(BYTES_IN_REG))) uniform float rslt[]
    ) 
{
  varying float sum = 0.0f;
  assume(((uniform uint64)((void*)x) & (32 * TARGET_WIDTH)-1) == 0);
  assume(((uniform uint64)((void*)y) & (32 * TARGET_WIDTH)-1) == 0);
  foreach ( i = 0 ... n ) { 
    varying float y_i = (y[i] * delta) + offset;
    // prefetch_l1(&(x[i+32])); // does not help
    // prefetch_l1(&(y[i+32])); // does not help
    sum += x[i] * y_i;
  }
  rslt[0] = reduce_add(sum);
}


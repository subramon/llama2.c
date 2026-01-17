#include "consts.h"
__attribute__((external_only))
export void 
div_s(
    __attribute__((aligned(BYTES_IN_REG))) uniform float x[] ,
    uniform float scalar,
    uniform int n
    ) 
{
  assume(n % programCount == 0);
  assume(((uniform uint64)((void*)x) & (32 * TARGET_WIDTH)-1) == 0);
  foreach ( i = 0 ... n ) {
    // prefetch_l1(&(x[i+32]));
    x[i] /= scalar;
  }
}

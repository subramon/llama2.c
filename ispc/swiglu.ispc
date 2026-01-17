#include "consts.h"
__attribute__((external_only))
export void
swiglu(
    __attribute__((aligned(BYTES_IN_REG))) uniform float x[], // [n]
    __attribute__((aligned(BYTES_IN_REG))) uniform float y[], // [n]
    uniform int n
    )
{
  assume(((uniform uint64)((void*)x) & (32 * TARGET_WIDTH)-1) == 0);
  assume(((uniform uint64)((void*)y) & (32 * TARGET_WIDTH)-1) == 0);
  assume(n % programCount == 0);
  foreach ( i = 0 ... n ) { 
    // prefetchw_l1(&(x[i+32]));
    varying float val = x[i];
    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    val *= (1.0f / (1.0f + exp(-val)));
    // elementwise multiply with y
    val *= y[i];
    x[i] = val;
  }
}

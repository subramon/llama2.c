#include "consts.h"
__attribute__((external_only))
export void 
rmsnorm(
    __attribute__((aligned(BYTES_IN_REG))) uniform float o[], 
    __attribute__((aligned(BYTES_IN_REG))) uniform float x[], 
    __attribute__((aligned(BYTES_IN_REG))) uniform float w[], 
    uniform int n
    ) 
{
  assume(((uniform uint64)((void*)o) & (32 * TARGET_WIDTH)-1) == 0);
  assume(((uniform uint64)((void*)x) & (32 * TARGET_WIDTH)-1) == 0);
  assume(((uniform uint64)((void*)w) & (32 * TARGET_WIDTH)-1) == 0);
  uniform int sz = (n / FLOATS_IN_REG ) * FLOATS_IN_REG;
  if ( sz != n ) { sz +=  FLOATS_IN_REG ; }
  assume(sz % programCount == 0);
  // cannot do assert inside ispc code 
  // calculate sum of squares
  varying float v_ss = 0.0f; // varying accumulator 
  foreach ( j = 0 ... sz ) {
    v_ss += x[j] * x[j];
  }
  uniform float u_ss = reduce_add(v_ss);  // uniform accumulator 
  u_ss /= n; // NOTE: Divide by n not sz 
  u_ss += 1e-5f;
  u_ss = 1.0f / sqrt(u_ss);
  // normalize and scale
  foreach ( j = 0 ... sz ) {
    o[j] = w[j] * x[j];
  }
  foreach ( j = 0 ... sz ) {
    o[j] *= u_ss;
  }
}

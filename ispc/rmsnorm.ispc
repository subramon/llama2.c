#include "consts.h"
export void 
rmsnorm(
    uniform float o[], 
    uniform float x[], 
    uniform float weight[], 
    uniform int n
    ) 
{
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
    o[j] = weight[j] * u_ss * x[j];
  }
}

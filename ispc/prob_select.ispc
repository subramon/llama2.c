#include "consts.h"
__attribute__((external_only))
export void
prob_select(
    __attribute__((aligned(BYTES_IN_REG))) uniform float p[], 
    uniform int n, 
    uniform float coin
    ) 
{
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  assume(((uniform uint64)((void*)p) & (32 * TARGET_WIDTH)-1) == 0);
  uniform float u_cdf = 0.0f;
  varying float v_cdf = 0.0f;
  foreach ( i = 0 ... n ) { 
  // TODO 
  }
  return n - 1; // in case of rounding errors
}

export void
prob_select(
    uniform float probabilities[], 
    uniform int n, 
    uniform float coin
    ) 
{
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  uniform float u_cdf = 0.0f;
  varying float v_cdf = 0.0f;
  foreach ( i = 0 ... n ) { 
  // TODO 
  }
  return n - 1; // in case of rounding errors
}

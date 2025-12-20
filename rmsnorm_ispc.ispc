export void 
rmsnorm(
    uniform float o[], 
    uniform float x[], 
    uniform float weight[], 
    uniform int n
    ) 
{
  // calculate sum of squares
  varying float v_ss = 0.0f; // varying accumulator 
  foreach ( j = 0 ... n ) {
    v_ss += x[j] * x[j];
  }
  uniform float u_ss = reduce_add(v_ss);  // uniform accumulator 
  u_ss /= n;
  u_ss += 1e-5f;
  u_ss = 1.0f / sqrt(u_ss);
  // normalize and scale
  foreach ( j = 0 ... n ) {
    o[j] = weight[j] * u_ss * x[j];
  }
}

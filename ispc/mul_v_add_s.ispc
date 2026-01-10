// x_i += scalar * y_i
export void 
mul_v_add_s(
    uniform float x[], 
    uniform float scalar,
    uniform float y[], 
    uniform int n
    ) 
{
  assume(n % programCount == 0);
  foreach ( i = 0 ... n ) { 
    // prefetch_l1(&(y[i+32]));
    // prefetchw_l1(&(x[i+32]));
    x[i] += scalar * y[i];
  }
}

// x_i <= x_i + scalar * y_i
export void 
mul_add_to(
    uniform float x[], 
    uniform float scalar,
    uniform float y[], 
    uniform int n
    ) 
{
  assume(n % programCount == 0);
  foreach ( i = 0 ... n ) { 
    x[i] += scalar * y[i];
  }
}

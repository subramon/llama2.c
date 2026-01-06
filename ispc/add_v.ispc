export void 
add_v(
    uniform float x[] ,
    uniform float y[], 
    uniform int n
    ) 
{
  assume(n % programCount == 0);
  foreach ( i = 0 ... n ) {
    x[i] += y[i];
  }
}

export void 
div_s(
    uniform float x[] ,
    uniform float scalar,
    uniform int n
    ) 
{
  assume(n % programCount == 0);
  foreach ( i = 0 ... n ) {
    // prefetch_l1(&(x[i+32]));
    x[i] /= scalar;
  }
}

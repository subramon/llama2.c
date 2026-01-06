export void 
div_s(
    uniform float x[] ,
    uniform float scalar,
    uniform int n
    ) 
{
  assume(n % programCount == 0);
  foreach ( i = 0 ... n ) {
    x[i] /= scalar;
  }
}

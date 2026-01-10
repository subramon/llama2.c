export void
swiglu(
    uniform float x[], // [n]
    uniform float y[], // [n]
    uniform int n
    )
{
  assume(n % programCount == 0);
  foreach ( i = 0 ... n ) { 
    // prefetchw_l1(&(x[i+32]));
    varying float val = x[i];
    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    val *= (1.0f / (1.0f + exp(-val)));
    // elementwise multiply with y
    val *= y[i];
    x[i] = val;
  }
}

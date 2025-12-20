export void 
dot_prod(
    uniform float x[], 
    uniform float y[], 
    uniform int n,
    uniform float rslt[]
    ) 
{
  varying float sum = 0.0f;
  foreach ( i = 0 ... n ) { 
    sum += x[i] * y[i];
  }
  rslt[0] = reduce_add(sum);
}


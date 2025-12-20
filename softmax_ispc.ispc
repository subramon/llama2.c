export void 
softmax(
    uniform float x[], 
    uniform int size
    ) 
{
  // find max value (for numerical stability)
  varying float v_max_val = 0; // TODO What should we set this to?
  foreach ( i = 0 ... size ) { 
    if (x[i] > v_max_val) {
      v_max_val = x[i];
    }
  }
  uniform float u_max_val = reduce_max(v_max_val);
  // exp and sum
  varying float v_sum = 0.0f;
  foreach ( i = 0 ... size ) { 
    x[i] = exp(x[i] - u_max_val);
    v_sum += x[i];
  }
  uniform float u_sum = reduce_add(v_sum);
  // normalize
  foreach ( i = 0 ... size ) { 
    x[i] /= u_sum;
  }
}

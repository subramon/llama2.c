extern void 
rmsnorm(
    float* restrict o, 
    const float* const x, 
    const float* const weight, 
    int n,
    int dummy // for parity with ISPC version
    ); 

extern void 
matmul_prefetch(
    float * restrict xout, 
    const float * const x,
    const float * const w,
    int n, 
    int d
    );

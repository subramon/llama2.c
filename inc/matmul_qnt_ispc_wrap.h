extern void 
matmul(
    float * restrict xout, 
    const float * const x,
    const uint8_t * const w,
    const float * const offset,
    const float * const delta,
    int n, 
    int d
    );

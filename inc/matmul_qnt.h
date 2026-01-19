extern void 
matmul_qnt(
    float * restrict xout, 
    const float * const x, 
    const float * const wf32, 
    const uint8_t * const wui8, 
    const float * const offset, 
    const float * const delta, 
    int n, 
    int d
    );

extern int
multi_head_attention(
    int nH,
    int pos,
    float *s_q, 
    float *s_att,
    float *s_kc,
    int head_size,
    int seq_len,
    int kv_dim,
    int kv_mul,
    int lidx
  );

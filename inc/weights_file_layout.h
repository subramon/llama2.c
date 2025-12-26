#ifndef __WEIGHTS_FILE_LAYOUT
#define __WEIGHTS_FILE_LAYOUT
typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

// listed in the order in which they appear in the weights file 
typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    size_t sz_tet;
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    size_t sz_rms_att;
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    size_t sz_wq;
    float* wk; // (layer, dim, n_kv_heads * head_size)
    size_t sz_wk;
    float* wv; // (layer, dim, n_kv_heads * head_size)
    size_t sz_wv;
    float* wo; // (layer, n_heads * head_size, dim)
    size_t sz_wo;
    float* rms_ffn_weight; // (layer, dim)
    // weights for ffn
    size_t sz_rms_ffn;
    float* w1; // (layer, hidden_dim, dim)
    size_t sz_w1;
    float* w2; // (layer, dim, hidden_dim)
    size_t sz_w2;
    float* w3; // (layer, hidden_dim, dim)
    size_t sz_w3;
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    size_t sz_rms_final;
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
    size_t sz_wcls;
} TransformerWeights;

typedef enum { 
  sp_token_embedding_table = 0,
  sp_rms_att_weight,
  sp_wq,
  sp_wk,
  sp_wv,
  sp_wo,
  sp_rms_ffn_weight,
  sp_w1,
  sp_w2,
  sp_w3,
  sp_rms_final_weight,
  sp_wcls,
  sp_num // place at end 
} splits;
#endif // __WEIGHTS_FILE_LAYOUT

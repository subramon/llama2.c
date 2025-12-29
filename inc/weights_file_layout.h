#ifndef __WEIGHTS_FILE_LAYOUT
#define __WEIGHTS_FILE_LAYOUT
typedef struct {
    int dim; // transformer dimension 
    int hidden_dim; // for ffn layers aka d_ff
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
    int head_size; // aka d_head DERIVED as dim/ n_heads
} Config;

// listed in the order in which they appear in the weights file 
typedef struct {
    // token embedding table 
    // The lookup table that maps token IDs to vectors.  
    // Shape: [vocab_size, dim]
    float* token_embedding_table;    // (vocab_size, dim)
    size_t sz_tet;
    // Transformer Layers (Repeated N times)
    // Each layer (e.g., layers.0., layers.1.) 
    // contains the following components:
    // =======================================================
    // [1] Attention Mechanism (Grouped Query Attention)
    // Llama 2 uses Grouped Query Attention (GQA).
    // attention.wq.weight: Query projections.
    float* wq; // (layer, dim, n_heads * head_size)
    size_t sz_wq;
    // attention.wk.weight: Key projections.
    float* wk; // (layer, dim, n_kv_heads * head_size)
    size_t sz_wk;
    // attention.wv.weight: Value projections.
    float* wv; // (layer, dim, n_kv_heads * head_size)
    size_t sz_wv;
    // attention.wo.weight: The output projection that merges heads back together.
    float* wo; // (layer, n_heads * head_size, dim)
    size_t sz_wo;
    // =======================================================
    // [3] Normalization
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    size_t sz_rms_att;
    // attention_norm.weight: RMSNorm applied before the attention block.
    //  ffn_norm.weight: RMSNorm applied before the feed-forward block.
    // weights for matmuls. note dim == n_heads * head_size
    float* rms_ffn_weight; // (layer, dim)
    size_t sz_rms_ffn;
    // =======================================================
    // [2] Feed-Forward Network (SwiGLU)
    // Unlike standard Transformers, 
    // Llama 2 uses three linear layers for its MLP:
    // feed_forward.w1.weight: 
    // Gate projection (used with the SiLU activation).
    float* w1; // (layer, hidden_dim, dim)
    size_t sz_w1;
    // feed_forward.w2.weight: 
    // Down-projection (brings the dimension back to dim).
    float* w2; // (layer, dim, hidden_dim)
    size_t sz_w2;
    // attention.w3.weight: Up-projection.
    float* w3; // (layer, hidden_dim, dim)
    size_t sz_w3;
    // =======================================================
    // [4] Final Output 
    // final rmsnorm after all layers 
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

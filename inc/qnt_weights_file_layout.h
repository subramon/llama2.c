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
} Config;
// WARNING! Do not add anything to the above struct. It will mess up
// the way Karpathy loads the checkpoint file

// listed in the order in which they appear in the weights file 
typedef struct {
    // token embedding table 
    // The lookup table that maps token IDs to vectors.  
    // Shape: [vocab_size, dim]
    uint8_t* token_embedding_table;    // (vocab_size, dim)
    size_t sz_tet;
    float *offset_tet; size_t sz_offset_tet;
    float *delta_tet;  size_t sz_delta_tet;
    // Transformer Layers (Repeated N times)
    // Each layer (e.g., layers.0., layers.1.) 
    // contains the following components:
    // =======================================================
    // [1] Attention Mechanism (Grouped Query Attention)
    // Llama 2 uses Grouped Query Attention (GQA).
    // attention.wq.weight: Query projections.
    uint8_t* wq; // (layer, dim, n_heads * head_size)
    size_t sz_wq;
    float *offset_wq; size_t sz_offset_wq;
    float *delta_wq;  size_t sz_delta_wq;
    // attention.wk.weight: Key projections.
    uint8_t* wk; // (layer, dim, n_kv_heads * head_size)
    size_t sz_wk;
    float *offset_wk; size_t sz_offset_wk;
    float *delta_wk;  size_t sz_delta_wk;
    // attention.wv.weight: Value projections.
    uint8_t* wv; // (layer, dim, n_kv_heads * head_size)
    size_t sz_wv;
    float *offset_wv; size_t sz_offset_wv;
    float *delta_wv;  size_t sz_delta_wv;
    // attention.wo.weight: The output projection that merges heads back together.
    uint8_t* wo; // (layer, n_heads * head_size, dim)
    size_t sz_wo;
    float *offset_wo; size_t sz_offset_wo;
    float *delta_wo;  size_t sz_delta_wo;
    // =======================================================
    // [3] Normalization
    uint8_t* rms_att_weight; // (layer, dim) rmsnorm weights
    size_t sz_rms_att;
    float *offset_rms_att; size_t sz_offset_rms_att;
    float *delta_rms_att;  size_t sz_delta_rms_att;
    // attention_norm.weight: RMSNorm applied before the attention block.
    //  ffn_norm.weight: RMSNorm applied before the feed-forward block.
    // weights for matmuls. note dim == n_heads * head_size
    uint8_t* rms_ffn_weight; // (layer, dim)
    size_t sz_rms_ffn;
    float *offset_rms_ffn; size_t sz_offset_rms_ffn;
    float *delta_rms_ffn;  size_t sz_delta_rms_ffn;
    // =======================================================
    // [2] Feed-Forward Network (SwiGLU)
    // Unlike standard Transformers, 
    // Llama 2 uses three linear layers for its MLP:
    // feed_forward.w1.weight: 
    // Gate projection (used with the SiLU activation).
    uint8_t* w1; // (layer, hidden_dim, dim)
    size_t sz_w1;
    float *offset_w1; size_t sz_offset_w1;
    float *delta_w1;  size_t sz_delta_w1;
    // feed_forward.w2.weight: 
    // Down-projection (brings the dimension back to dim).
    uint8_t* w2; // (layer, dim, hidden_dim)
    size_t sz_w2;
    float *offset_w2; size_t sz_offset_w2;
    float *delta_w2;  size_t sz_delta_w2;
    // attention.w3.weight: Up-projection.
    uint8_t* w3; // (layer, hidden_dim, dim)
    size_t sz_w3;
    float *offset_w3; size_t sz_offset_w3;
    float *delta_w3;  size_t sz_delta_w3;
    // =======================================================
    // [4] Final Output 
    // final rmsnorm after all layers 
    uint8_t* rms_final_weight; // (dim,)
    size_t sz_rms_final;
    float *offset_rms_final; size_t sz_offset_rms_final;
    float *delta_rms_final;  size_t sz_delta_rms_final;
    // (optional) classifier weights for the logits, on the last layer
    // wcls (weights for the classification or unembedding layer) tensor 
    // in Llama 2 inference has the shape (vocab_size, hidden_size).

    uint8_t* wcls;
    size_t sz_wcls;
    float *offset_wcls; size_t sz_offset_wcls;
    float *delta_wcls;  size_t sz_delta_wcls;
} TransformerWeights;
#endif // __WEIGHTS_FILE_LAYOUT

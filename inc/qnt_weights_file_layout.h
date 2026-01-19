#ifndef __QNT_WEIGHTS_FILE_LAYOUT
#define __QNT_WEIGHTS_FILE_LAYOUT
// listed in the order in which they appear in the weights file 
typedef struct {
    // token embedding table 
    // The lookup table that maps token IDs to vectors.  
    uint8_t *qnt_tet;  size_t sz_qnt_tet; // (vocab_size, dim)
    float *offset_tet; size_t sz_offset_tet;
    float *delta_tet;  size_t sz_delta_tet;
    // Transformer Layers (Repeated N times)
    // Each layer (e.g., layers.0., layers.1.) 
    // contains the following components:
    // =======================================================
    // [1] Attention Mechanism (Grouped Query Attention)
    // Llama 2 uses Grouped Query Attention (GQA).
    // attention.wq.weight: Query projections.
    uint8_t* qnt_wq; size_t sz_qnt_wq; 
    // (layer, dim, n_heads * head_size)
    float *offset_wq; size_t sz_offset_wq;
    float *delta_wq;  size_t sz_delta_wq;

    // attention.wk.weight: Key projections.
    uint8_t* qnt_wk; size_t sz_qnt_wk; 
    // (layer, dim, n_kv_heads * head_size)
    float *offset_wk; size_t sz_offset_wk;
    float *delta_wk;  size_t sz_delta_wk;

    // attention.wv.weight: Value projections.
    uint8_t* qnt_wv; size_t sz_qnt_wv;
    // (layer, dim, n_kv_heads * head_size)
    float *offset_wv; size_t sz_offset_wv;
    float *delta_wv;  size_t sz_delta_wv;

    // attention.wo.weight: 
    // The output projection that merges heads back together.
    uint8_t* qnt_wo; size_t sz_qnt_wo;
    // (layer, n_heads * head_size, dim)
    float *offset_wo; size_t sz_offset_wo;
    float *delta_wo;  size_t sz_delta_wo;
    // =======================================================
    // [3] Normalization
    // attention_norm.weight: RMSNorm applied before the attention block.
    uint8_t* qnt_rms_att_weight; size_t sz_qnt_rms_att;
    // (layer, dim) rmsnorm weights
    float *offset_rms_att; size_t sz_offset_rms_att;
    float *delta_rms_att;  size_t sz_delta_rms_att;

    //  ffn_norm.weight: RMSNorm applied before the feed-forward block.
    // weights for matmuls. note dim == n_heads * head_size
    uint8_t* qnt_rms_ffn_weight; size_t sz_qnt_rms_ffn;
    // (layer, dim)
    float *offset_rms_ffn; size_t sz_offset_rms_ffn;
    float *delta_rms_ffn;  size_t sz_delta_rms_ffn;
    // =======================================================
    // [2] Feed-Forward Network (SwiGLU)
    // Unlike standard Transformers, 
    // Llama 2 uses three linear layers for its MLP:
    // feed_forward.w1.weight: 
    // Gate projection (used with the SiLU activation).
    uint8_t* qnt_w1; size_t sz_qnt_w1;
    // (layer, hidden_dim, dim)
    float *offset_w1; size_t sz_offset_w1;
    float *delta_w1;  size_t sz_delta_w1;

    // feed_forward.w2.weight: 
    // Down-projection (brings the dimension back to dim).
    uint8_t* qnt_w2; size_t sz_qnt_w2;
    // (layer, dim, hidden_dim)
    float *offset_w2; size_t sz_offset_w2;
    float *delta_w2;  size_t sz_delta_w2;

    // attention.w3.weight: Up-projection.
    uint8_t* qnt_w3; size_t sz_qnt_w3;
    // (layer, hidden_dim, dim)
    float *offset_w3; size_t sz_offset_w3;
    float *delta_w3;  size_t sz_delta_w3;
    // =======================================================
    // [4] Final Output 
    // final rmsnorm after all layers 
    uint8_t* qnt_rms_final_weight; size_t sz_qnt_rms_final;
    // (dim,)
    float *offset_rms_final; size_t sz_offset_rms_final;
    float *delta_rms_final;  size_t sz_delta_rms_final;

    // (optional) classifier weights for the logits, on the last layer
    // wcls (weights for the classification or unembedding layer) tensor 
    // in Llama 2 inference has the shape (vocab_size, hidden_size).

    uint8_t* qnt_wcls; size_t sz_qnt_wcls;
    // (vocab_size, hidden_size).
    float *offset_wcls; size_t sz_offset_wcls;
    float *delta_wcls;  size_t sz_delta_wcls;
} QntTransformerWeights;
#endif // __QNT_WEIGHTS_FILE_LAYOUT

/* Inference for Llama-2 Transformer model in pure C */
#include <stdio.h>
#include <stdbool.h>
#include <omp.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <x86intrin.h> // for rdtsc


#include "consts.h"
#include "macros.h"
#include "matmul.h"
#include "rmsnorm.h"
#include "read_config.h"
#include "softmax.h"
#include "mmap_weights.h"
#include "qnt_mmap_weights.h"
#include "run_state.h"
#include "rope.h"
#include "dot_prod.h"
#include "add_v.h"
#include "div_s.h"
#include "mul_v_add_s.h"
#include "swiglu.h"
#include "argmax.h"
#include "prob_select.h"
#include "target_width.h"

bool g_quantize; 
uint64_t g_t_prefetch;
uint64_t g_n_prefetch;
uint64_t g_t_expt;
uint64_t g_n_expt;
uint64_t g_t_matmul;
uint64_t g_n_matmul;
uint64_t g_t_omp_loop;
uint64_t g_n_omp_loop;
uint64_t g_t_rmsnorm;
uint64_t g_t_softmax;
uint64_t g_t_dot_prod;
uint64_t g_t_add_v;
uint64_t g_t_div_s;
uint64_t g_t_mul_v_add_s;
uint64_t g_t_swiglu;
uint64_t g_t_argmax;
uint64_t g_t_prob_select;
// -------------------------------------------------------------------
// Transformer model
#include "weights_file_layout.h"
#include "qnt_weights_file_layout.h"
// State
#include "run_state.h"

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    QntTransformerWeights qnt_weights; // the quantized weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
} Transformer;

static int
read_checkpoint(
    const char * const checkpoint,  // file containing checkpoint 
    Config* ptr_config, 
    TransformerWeights* ptr_weights,
    QntTransformerWeights* ptr_qnt_weights
    ) 
{
  int status = 0; 

  status = read_config(checkpoint, ptr_config); cBYE(status);
  // negative vocab size is hacky way of signaling unshared weights. 
  // bit yikes. TODO P3 
  int shared_weights = ptr_config->vocab_size > 0 ? 1 : 0;
  ptr_config->vocab_size = abs(ptr_config->vocab_size);
  status = mmap_weights(ptr_weights); cBYE(status);
  if ( g_quantize ) { 
    status = qnt_mmap_weights(ptr_qnt_weights); cBYE(status);
  }
BYE:
  return status;
}

static int
build_transformer(
    Transformer *t, 
    char* checkpoint_path
    ) 
{
  int status = 0;
  // read in the Config and the Weights from the checkpoint
  status = read_checkpoint(checkpoint_path, 
      &(t->config), &(t->weights), &(t->qnt_weights));
  cBYE(status);
  // allocate the RunState buffers
  status = malloc_run_state(&t->state, &t->config); cBYE(status);
BYE:
  return status;
}

static void 
free_transformer(
    Transformer* t
    ) 
{
  free_run_state(&t->state);
  munmap_weights(&t->weights);
}

// ---------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer


// This function updates w->state->logits
static int
forward(
    Transformer* transformer, 
    int token, 
    int pos
    ) 
{
  int status = 0;
  // a few convenience variables
  Config* p = &transformer->config;
  TransformerWeights* w = &transformer->weights;
  QntTransformerWeights* qw = &transformer->qnt_weights;
  RunState* s = &transformer->state;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim =  p->hidden_dim;
  int head_size = dim / p->n_heads;
  size_t ispc_dim = mcr_round_up(p->dim);
  size_t ispc_hidden_dim = mcr_round_up(p->hidden_dim);
  size_t ispc_seq_len = mcr_round_up(p->seq_len);
  size_t ispc_kv_dim = mcr_round_up(kv_dim);
  size_t ispc_head_size = mcr_round_up(head_size);

  uint64_t t1, t2;

#ifdef DEBUG
  if ( ( pos < 0 ) || ( pos >= p->seq_len ) ) { go_BYE(-1); }
  if ( ( token < 0 ) || ( token >= p->vocab_size ) ) { go_BYE(-1); }
#endif
  // copy the token embedding into x
  float* tet_ptr = mcr_2d_to_1d(w->token_embedding_table, token, ispc_dim);
  memcpy(x, tet_ptr, ((size_t)dim*sizeof(float)));

  // forward all the layers
  for ( int l = 0; l < p->n_layers; l++) {
    // attention rmsnorm
    float * const w_rms_att_l = mcr_2d_to_1d(w->rms_att_weight, l, ispc_dim);
    rmsnorm(s->xb, x, w_rms_att_l, dim);

    // key_ptr and val_ptr point to appropriate location in kv cache
    float *key_ptr = mcr_3d_to_1d(s->kc, l, pos, p->seq_len, ispc_kv_dim);
    float *val_ptr = mcr_3d_to_1d(s->vc, l, pos, p->seq_len, ispc_kv_dim);

    float * const w_q = mcr_3d_to_2d(w->wq, l, dim, ispc_dim);
    float * const w_k = mcr_3d_to_2d(w->wk, l, dim, ispc_kv_dim);
    float * const w_v = mcr_3d_to_2d(w->wv, l, dim, ispc_kv_dim);

    if ( g_quantize ) { 
      float * const qw_q = mcr_3d_to_2d(qw->qnt_wq, l, dim, ispc_dim);
      float * const qw_k = mcr_3d_to_2d(qw->qnt_wk, l, dim, ispc_kv_dim);
      float * const qw_v = mcr_3d_to_2d(qw->qnt_wv, l, dim, ispc_kv_dim);

      float * const qw_q_off = mcr_2d_to_1d(qw->offset_wq, l, ispc_dim);
      float * const qw_k_off = mcr_2d_to_1d(qw->offset_wk, l, ispc_dim);
      float * const qw_v_off = mcr_2d_to_1d(qw->offset_wv, l, ispc_dim);

      float * const qw_q_del = mcr_2d_to_1d(qw->delta_wq, l, ispc_dim);
      float * const qw_k_del = mcr_2d_to_1d(qw->delta_wk, l, ispc_dim);
      float * const qw_v_del = mcr_2d_to_1d(qw->delta_wv, l, ispc_dim);
      // qkv matmuls for this position
      matmul_qnt(s->q,    s->xb, w_q, qw_q, qw_q_off, qw_q_del, dim, dim);
      matmul_qnt(key_ptr, s->xb, w_k, qw_k, qw_k_off, qw_k_del, dim, kv_dim);
      matmul_qnt(val_ptr, s->xb, w_v, qw_v, qw_v_off, qw_v_del, dim, kv_dim);
    }
    else {
      // qkv matmuls for this position
      /* EXPERIMENTAL 
         matmul_prefetch(s->q,    s->xb, w_q, dim, dim);
         matmul_prefetch(s->q,    s->xb, w_q, dim, dim);
         matmul_prefetch(key_ptr, s->xb, w_k, dim, kv_dim);
         */
      uint64_t t0 = __rdtsc();

      matmul(s->q,    s->xb, w_q, dim, dim); 
      matmul(key_ptr, s->xb, w_k, dim, kv_dim); 
      matmul(val_ptr, s->xb, w_v, dim, kv_dim); 

      g_n_expt += 
        2*dim*dim + 
        2*dim*kv_dim + 
        2*dim*kv_dim;
      g_t_expt += __rdtsc() - t0;
    }

    // RoPE relative positional encoding: 
    // complex-valued rotate q and k in each head
    status = rope(dim, kv_dim, head_size, pos, s->q, key_ptr); cBYE(status);

    // multihead attention. iterate over all heads in parallel
    // TODO P3: Study taskloop in OpenMP
    // TODO P3: Consider Collapse these 2 loops into one
    uint64_t t_start =  __rdtsc();
    // CAUTION: Parallelizing this loop slows things down for gcc
    // STRANGE: Slows it down for gcc but not for ISPC. Puzzling...
    // However, in one simple case, speedup was only 1.5x
    // I don't think there is enough work to justify overhead of omp
#pragma omp parallel for 
    for ( int h = 0; h < p->n_heads; h++) {
      // get the query vector for this head
      const float* const q_h = mcr_2d_to_1d(s->q, h, ispc_head_size); 
      // attention scores for this head
      float* att_h = mcr_2d_to_1d(s->att, h, ispc_seq_len);
      // iterate over all timesteps, including the current one
      for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        float *keyptr = mcr_3d_to_1d(s->kc, l, t, p->seq_len, ispc_kv_dim);
        keyptr += (h / kv_mul) * head_size;
#ifdef DEBUG
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float *old_k = s->kc + loff + t * kv_dim + (h / kv_mul) * head_size;
        if ( old_k != keyptr ) { status = -1; continue; }
#endif
        // calculate the attention score as the dot product of q and k
        float score;
        dot_prod(q_h, keyptr, head_size, &score); 
        score /= sqrtf((float)head_size);
        // save the score to the attention buffer
        att_h[t] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax(att_h, pos + 1);

      // weighted sum of the values, store back into xb
      float* const xb = s->xb + h * head_size;
      memset(xb, 0, (size_t)(head_size * sizeof(float)));
      for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        float *valptr = mcr_3d_to_1d(s->vc, l, t, p->seq_len, ispc_kv_dim);
        valptr += (h / kv_mul) * head_size;
#ifdef DEBUG
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float *old_v = s->vc + loff + t * kv_dim + (h / kv_mul) * head_size;
        if ( old_v != valptr ) { WHEREAMI; status = -1; continue; }
#endif
        // get the attention weight for this timestep
        float a = att_h[t];
        // accumulate the weighted value into xb
        mul_v_add_s(xb, a, valptr, head_size); // xb[i] += a * v[i]
      }
    }
    g_t_omp_loop += _rdtsc() - t_start;
    g_n_omp_loop ++;
    cBYE(status);

    // final matmul to get the output of the attention
    float *wo_ptr = mcr_3d_to_2d(w->wo, l, dim, ispc_dim);
    matmul(s->xb2, s->xb, wo_ptr, dim, dim);

    // residual connection back into x
    add_v(x, s->xb2, dim); 

    // ffn rmsnorm
    float *rms_ffn_l = mcr_2d_to_1d(w->rms_ffn_weight, l, ispc_dim);
    rmsnorm(s->xb, x, rms_ffn_l, dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    float *w1_ptr = mcr_3d_to_2d(w->w1, l, hidden_dim, ispc_dim);
    float *w3_ptr = mcr_3d_to_2d(w->w3, l, hidden_dim, ispc_dim);
    matmul(s->hb,  s->xb, w1_ptr, dim, hidden_dim);
    matmul(s->hb2, s->xb, w3_ptr, dim, hidden_dim);

    // SwiGLU non-linearity
    swiglu(s->hb, s->hb2, hidden_dim);

    // final matmul to get the output of the ffn
    float *w2_ptr = mcr_3d_to_2d(w->w2, l, dim, ispc_hidden_dim);
    matmul(s->xb, s->hb, w2_ptr, hidden_dim, dim);

    // residual connection
    add_v(x, s->xb, dim); 
  }

  // final rmsnorm
  rmsnorm(x, x, w->rms_final_weight, dim);

  // classifier into logits TODO use pointer for wcls 
  matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
BYE:
  return status;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

static int 
compare_tokens(
    const void *a, 
    const void *b
    ) 
{
  return strcmp(((const TokenIndex*)a)->str, ((const TokenIndex*)b)->str);
}

static void 
build_tokenizer(
    Tokenizer* t, 
    const char* const tokenizer_path, 
    int vocab_size
    ) 
{
  // i should have written the vocab_size into the tokenizer file... sigh
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (char**)malloc(vocab_size * sizeof(char*));
  t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; // initialized lazily
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }
  // read in the file
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
  int len;
  for (int i = 0; i < vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
    if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    t->vocab[i] = (char *)malloc((size_t)len + 1);
    if (fread(t->vocab[i], (size_t)len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    t->vocab[i][len] = '\0'; // add the string terminating token
  }
  fclose(file);
}

static void 
free_tokenizer(
    Tokenizer* t
    ) 
{
  for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

static char* 
decode(
    Tokenizer* t, 
    int prev_token, 
    int token
    )
{
  char *piece = t->vocab[token];
  // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
  if ( ( prev_token == 1 ) && ( piece[0] == ' ' ) ) { piece++; }
  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char*)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

static void safe_printf(
    char *piece
    ) 
{
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  if (piece == NULL) { return; }
  if (piece[0] == '\0') { return; }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  printf("%s", piece);
}

static int 
str_lookup(
    const char *str, 
    TokenIndex *sorted_vocab, 
    int vocab_size
    ) 
{
  // efficiently find the perfect match for str in vocab, return its index or -1 if not found
  TokenIndex tok = { .str = str }; // acts as the key to search for
  TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

static int
encode(
    Tokenizer* t, 
    char *text, 
    int8_t bos, 
    int8_t eos, 
    int *tokens, 
    int *n_tokens
    ) 
{
  int status = 0;
  char* str_buffer = NULL;
  // encode the string text (input) into an upper-bound preallocated tokens[] array
  // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
  if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

  if (t->sorted_vocab == NULL) {
    // lazily malloc and sort the vocabulary
    t->sorted_vocab = malloc((size_t)t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, (size_t)t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // create a temporary buffer that will store merge candidates of always two consecutive tokens
  // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
  str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
  return_if_malloc_failed(str_buffer);
  size_t str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=1) token, if desired
  if (bos) tokens[(*n_tokens)++] = 1;

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have the
  // energy to read more of the sentencepiece code to figure out what it's doing
  if (text[0] != '\0') {
    int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
    tokens[(*n_tokens)++] = dummy_prefix;
  }

  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ↔ UTF-8 conversion
  // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
  // U+0000	U+007F	    0xxxxxxx
  // U+0080	U+07FF	    110xxxxx	10xxxxxx
  // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
  // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

  // process the raw (UTF-8) byte sequence of the input string
  for (char *c = text; *c != '\0'; c++) {

    // reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
    // 0x80 is 10000000
    // in UTF-8, all continuation bytes start with "10" in first two bits
    // so in English this is: "if this byte is not a continuation byte"
    if ((*c & 0xC0) != 0x80) {
      // this byte must be either a leading byte (11...) or an ASCII char (0x...)
      // => reset our location, as we're starting a new UTF-8 codepoint
      str_len = 0;
    }

    // append the current byte to the buffer
    str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
    str_buffer[str_len] = '\0';

    // while the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning str_buffer size.
    if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // ok c+1 is not a continuation byte, so we've read in a full codepoint
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

    if (id != -1) {
      // we found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      for ( size_t i=0; i < str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair each iteration, according the scores in vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (int i=0; i < (*n_tokens-1); i++) {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1) {
      break; // we couldn't find any more pairs to merge, so we're done
    }

    // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    tokens[best_idx] = best_id;
    // delete token at position best_idx+1, shift the entire sequence back 1
    for (int i = best_idx+1; i < (*n_tokens-1); i++) {
      tokens[i] = tokens[i+1];
    }
    (*n_tokens)--; // token length decreased
  }

  // add optional EOS (=2) token, if desired
  if (eos) tokens[(*n_tokens)++] = 2;

BYE:
  free_if_non_null(str_buffer);
  return status;
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

static int 
compare(
    const void* a, 
    const void* b
    ) 
{
  const ProbIndex* a_ = (const ProbIndex*) a;
  const ProbIndex* b_ = (const ProbIndex*) b;
  if (a_->prob > b_->prob) return -1;
  if (a_->prob < b_->prob) return 1;
  return 0;
}

static int sample_topp(
    float* probabilities, 
    int n, 
    float topp, 
    ProbIndex* probindex, 
    float coin
    ) 
{
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / ((float)n - 1.0f);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, (size_t)n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

static void 
build_sampler(
    Sampler* sampler, 
    int vocab_size, 
    float temperature, 
    float topp, 
    unsigned long long rng_seed
    ) 
{
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = malloc((size_t)sampler->vocab_size * sizeof(ProbIndex));
}

static void free_sampler(
    Sampler* sampler
    ) 
{
  free(sampler->probindex);
}

static unsigned int 
random_u32(
    unsigned long long *state
    ) 
{
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (uint32_t)((*state * 0x2545F4914F6CDD1Dull) >> 32);
}
static float random_f32(
    unsigned long long *state
    ) 
{ // random float32 in [0,1)
  return ((float)(random_u32(state) >> 8)) / 16777216.0f;
}

static int 
sample(
    Sampler* sampler, 
    float* logits
    ) 
{
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = argmax(logits, sampler->vocab_size);
  } 
  else {
    // apply the temperature to the logits
    div_s(logits, sampler->temperature, sampler->vocab_size);
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = prob_select(logits, sampler->vocab_size, coin);
    } 
    else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

// ----------------------------------------------------------------------------
// utilities: time

static long 
time_in_ms(
    void
    ) 
{
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

static int
generate(
    Transformer *transformer, 
    Tokenizer *tokenizer, 
    Sampler *sampler, 
    char *prompt, 
    int n_steps
    ) 
{
  int status = 0;
  int* prompt_tokens = NULL; int num_prompt_tokens = 0;
  const char *empty_prompt = "";
  if ( prompt == NULL) { prompt = empty_prompt; }

  // encode the (string) prompt into tokens sequence
  prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  status = encode(tokenizer, prompt, 1, 0, prompt_tokens, 
      &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    go_BYE(-1); 
  }

  // start the main loop
  long start = 0;  // used to time our code, only initialized after first iteration
  int next;        // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;     // position in the sequence
  while (pos < n_steps) {

    // forward the transformer to get logits for the next token
    status = forward(transformer, token, pos); cBYE(status);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos + 1];
    } 
    else {
      // otherwise sample the next token from the logits
      next = sample(sampler, transformer->state.logits);
    }
    pos++;

    // data-dependent terminating condition: the BOS (=1) token delimits sequences
    if (next == 1) { break; }

    // print the token as string, decode it with the Tokenizer object
    char* piece = decode(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) { start = time_in_ms(); }
  }
  printf("\n");

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  if (pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
  }

BYE:
  free_if_non_null(prompt_tokens);
  return status;
}

static void 
read_stdin(
    const char* guide, 
    char* buffer, 
    size_t bufsize
    ) 
{
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, (int)bufsize, stdin) != NULL) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0'; // strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

static int
chat(
    Transformer *transformer, 
    Tokenizer *tokenizer, 
    Sampler *sampler, 
    char *cli_user_prompt, 
    char *cli_system_prompt, 
    int steps
    ) 
{
  int status = 0;

  // buffers for reading the system prompt and user prompt from stdin
  // you'll notice they are somewhat haphazardly and unsafely set atm
  char system_prompt[512];
  char user_prompt[512];
  char rendered_prompt[1152];
  int num_prompt_tokens = 0;
  int* prompt_tokens = NULL;
  int user_idx;

  prompt_tokens = malloc(1152 * sizeof(int)); // TODO P3 Why 1152?
  return_if_malloc_failed(prompt_tokens); 

  // start the main loop
  int8_t user_turn = 1; // user starts
  int next;        // will store the next token in the sequence
  int token;       // stores the current token to feed into the transformer
  int pos = 0;     // position in the sequence
  while (pos < steps) {

    // when it is the user's turn to contribute tokens to the dialog...
    if (user_turn) {
      // get the (optional) system prompt at position 0
      if (pos == 0) {
        // at position 0, the user can also contribute a system prompt
        if (cli_system_prompt == NULL) {
          // system prompt was not passed in, attempt to get it from stdin
          read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
        } else {
          // system prompt was passed in, use it
          strcpy(system_prompt, cli_system_prompt);
        }
      }
      // get the user prompt
      if (pos == 0 && cli_user_prompt != NULL) {
        // user prompt for position 0 was passed in, use it
        strcpy(user_prompt, cli_user_prompt);
      } else {
        // otherwise get user prompt from stdin
        read_stdin("User: ", user_prompt, sizeof(user_prompt));
      }
      // render user/system prompts into the Llama 2 Chat schema
      if (pos == 0 && system_prompt[0] != '\0') {
        char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
        sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
      } else {
        char user_template[] = "[INST] %s [/INST]";
        sprintf(rendered_prompt, user_template, user_prompt);
      }
      // encode the rendered prompt into tokens
      encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
      user_idx = 0; // reset the user index
      user_turn = 0;
      printf("Assistant: ");
    }

    // determine the token to pass into the transformer next
    if (user_idx < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt token
      token = prompt_tokens[user_idx++];
    } else {
      // otherwise use the next token sampled from previous turn
      token = next;
    }
    // EOS (=2) token ends the Assistant turn
    if (token == 2) { user_turn = 1; }

    // forward the transformer to get logits for the next token
    status = forward(transformer, token, pos); cBYE(status);
    next = sample(sampler, transformer->state.logits);
    pos++;

    if (user_idx >= num_prompt_tokens && next != 2) {
      // the Assistant is responding, so print its output
      char* piece = decode(tokenizer, token, next);
      safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
      fflush(stdout);
    }
    if (next == 2) { printf("\n"); }
  }
  printf("\n");
  free(prompt_tokens);
BYE:
  return status;
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

static void 
error_usage(
    void
    )

{
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  exit(EXIT_FAILURE);
}

int main(
    int argc, 
    char *argv[]
    ) 
{
  int status = 0;
  g_quantize = false;
  g_t_matmul = 0;
  g_t_rmsnorm = 0;
  g_t_softmax = 0;
  g_t_dot_prod = 0;
  g_t_add_v = 0;
  g_t_div_s = 0;
  g_t_mul_v_add_s = 0;
  g_t_swiglu = 0;
  g_t_argmax = 0;
  g_t_prob_select = 0;
  // default parameters
  char *checkpoint_path = NULL;  // e.g. out/model.bin
  const char *tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 256;            // number of steps to run for
  char *prompt = NULL;        // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  const char *mode = "generate";    // generate|chat
  char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

  // TODO P4: These should not be necessary. 
  // Doint it because I am Having problem with #define in ISPC
  {
    int x; 
    target_width(&x);
    if ( FLOATS_IN_REG != x ) { go_BYE(-1); }
    if ( BYTES_IN_REG != (sizeof(float) * FLOATS_IN_REG) ) { go_BYE(-1); }
  }
  omp_set_num_threads(16);
  printf("nP = %d\n", omp_get_num_procs());

  // poor man's C argparse so we can override the defaults above from the command line
  if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
  for (int i = 2; i < argc; i+=2) {
    // do some basic validation
    if (i + 1 >= argc) { error_usage(); } // must have arg after flag
    if (argv[i][0] != '-') { error_usage(); } // must start with dash
    if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
    // read in the args
    if (argv[i][1] == 't') { temperature = (float)atof(argv[i + 1]); }
    else if (argv[i][1] == 'p') { topp = (float)atof(argv[i + 1]); }
    else if (argv[i][1] == 's') { rng_seed = (unsigned long long)atoi(argv[i + 1]); }
    else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
    else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
    else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
    else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
    else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
    else if (argv[i][1] == 'q') { // for quantization
      if ( strcasecmp(argv[i+1], "true") == 0 ) { 
        g_quantize = true;
      }
      else if ( strcasecmp(argv[i+1], "false") == 0 ) { 
        g_quantize = false;
      }
      else {
        go_BYE(-1);
      }
    }
    else { error_usage(); }
  }

  // parameter validation/overrides
  if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0) temperature = 0.0f;
  if (topp < 0.0 || 1.0 < topp) topp = 0.9f;
  if (steps < 0) steps = 0;

  // build the Transformer via the model .bin file
  Transformer transformer;
  memset(&transformer.config, 0, sizeof(transformer.config));
  memset(&transformer.weights, 0, sizeof(transformer.weights));
  memset(&transformer.qnt_weights, 0, sizeof(transformer.qnt_weights));
  memset(&transformer.state, 0, sizeof(transformer.state));

  status = build_transformer(&transformer, checkpoint_path); cBYE(status);
  if ( (steps == 0) || (steps > transformer.config.seq_len) ) {
    steps = transformer.config.seq_len; // override to ~max length
  }

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  // run!
  if (strcmp(mode, "generate") == 0) {
    uint64_t t1 = __rdtsc();
    status = generate(&transformer, &tokenizer, &sampler, prompt, steps);
    uint64_t t2 = __rdtsc();
    printf("Total  clocks   = %" PRIu64 "\n", t2 -t1);

    printf("expt   clocks   = %" PRIu64 "\n", g_t_expt); 
    printf("expt   flops    = %" PRIu64 "\n", g_n_expt); 
    printf("expt   Gflops/s = %lf\n", g_n_expt*5.1/g_t_expt);

    printf("matmul clocks   = %" PRIu64 "\n", g_t_matmul); 
    printf("matmul flops    = %" PRIu64 "\n", g_n_matmul); 
    printf("matmul Gflops/s = %lf\n", g_n_matmul*5.1/g_t_matmul);

    printf("omp    clocks   = %" PRIu64 "\n", g_t_omp_loop); 
    printf("omp    loops    = %" PRIu64 "\n", g_n_omp_loop); 

    printf("prefetch    clocks   = %" PRIu64 "\n", g_t_prefetch); 
    printf("prefetch    calls    = %" PRIu64 "\n", g_n_prefetch); 

    printf("dot_prod clocks = %" PRIu64 "\n", g_t_dot_prod); 
  } 
  else if (strcmp(mode, "chat") == 0) {
    status = chat(&transformer, &tokenizer, &sampler, prompt, 
        system_prompt, steps);
  } 
  else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }
  cBYE(status);

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
BYE:
  return status;
}
#endif

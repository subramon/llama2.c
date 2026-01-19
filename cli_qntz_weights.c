#include <stdio.h>
#include "rs_mmap.h"
#include "consts.h"
#include "macros.h"
#include "weights_file_layout.h"
#include "read_config.h"
#include "qntz_2d.h"
#include "qntz_3d.h"
#include "mmap_weights.h"

int
main(
    int argc,
    char **argv
    )
{
  int status = 0;
  Config C; memset(&C, 0, sizeof(Config));
  const char *wfile; // input weights fle 
  if ( argc != 2 ) { go_BYE(-1); }
  wfile = argv[1];
  //---------------------------------------------------
  status = read_config(wfile, &C); cBYE(status); 
  int head_size = C.dim / C.n_heads;
  //---------------------------------------------------
  int padding = 0; float fzero = 0;
  // Create individual file for each "split"
  // create token_embedding_table
  status = qntz_2d(
      "_token_embedding_table.bin", "_token_embedding_table.ui8",
      "_token_embedding_table.offset", "_token_embedding_table.delta", 
      C.vocab_size, C.dim);
  cBYE(status);
  //-------------------------------------------------------
  // create rms_att_weight
  status = qntz_2d(
      "_rms_att_weight.bin", "_rms_att_weight.ui8", 
      "_rms_att_weight.offset", "_rms_att_weight.delta", 
      C.n_layers, C.dim);
  cBYE(status);
  //-------------------------------------------------------
  // create wq
  status = qntz_3d(
      "_wq.bin", "_wq.ui8", 
      "_wq.offset", "_wq.delta", 
      C.n_layers, C.dim, C.n_heads * head_size);
  cBYE(status);
  //-------------------------------------------------------
  // create wk
  status = qntz_3d(
      "_wk.bin", "_wk.ui8", "_wk.offset", "_wk.delta", 
      C.n_layers, C.dim, C.n_kv_heads * head_size);
  cBYE(status);
  //-------------------------------------------------------
  // create wv
  status = qntz_3d(
      "_wv.bin", "_wv.ui8", "_wv.offset", "_wv.delta", 
      C.n_layers, C.dim, C.n_kv_heads * head_size);
  cBYE(status);
  //-------------------------------------------------------
  // create wo
  status = qntz_3d(
      "_wo.bin", "_wo.ui8", "_wo.offset", "_wo.delta", 
      C.n_layers, C.n_heads * head_size, C.dim);
  cBYE(status);
  //-------------------------------------------------------
  // create rms_ffn_weight
  status = qntz_2d(
      "_rms_ffn_weight.bin", "_rms_ffn_weight.ui8", 
      "_rms_ffn_weight.offset", "_rms_ffn_weight.delta", 
      C.n_layers, C.dim);
  cBYE(status);
  //-------------------------------------------------------
  // create w1
  status = qntz_3d(
      "_w1.bin", "_w1.ui8", "_w1.offset", "_w1.delta", 
      C.n_layers, C.dim, C.hidden_dim);
  cBYE(status);
  //-------------------------------------------------------
  // create w2
  status = qntz_3d(
      "_w2.bin", "_w2.ui8", "_w2.offset", "_w2.delta", 
      C.n_layers, C.hidden_dim, C.dim);
  cBYE(status);
  //-------------------------------------------------------
  // create w3
  status = qntz_3d(
      "_w3.bin", "_w3.ui8", "_w3.offset", "_w3.delta", 
      C.n_layers, C.dim, C.hidden_dim);
  cBYE(status);
  //-------------------------------------------------------
BYE:
  return status;
}

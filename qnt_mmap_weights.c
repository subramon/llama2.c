#include <stdio.h>
#include "macros.h"
#include "rs_mmap.h"
#include "consts.h"
#include "qnt_weights_file_layout.h"
#include "qnt_mmap_weights.h"

int
qnt_mmap_weights(
    TransformerWeights * restrict ptr_w
    )
{
  int status = 0;
  char *X = NULL; size_t nX = 0;
  // create token_embedding_table
  status = rs_mmap("_token_embedding_table.bin", &X, &nX, 0); cBYE(status);
  ptr_w->token_embedding_table = (uint8_t *)X; X = NULL;
  ptr_w->sz_tet = nX; nX = 0;
  //-------------------------------------------------------
  // create rms_att_weight
  status = rs_mmap("_rms_att_weight.bin", &X, &nX, 0); cBYE(status);
  ptr_w->rms_att_weight = (uint8_t *)X; X = NULL;
  ptr_w->sz_rms_att = nX; nX = 0;
  //-------------------------------------------------------
  // create wq
  status = rs_mmap("_wq.bin", &X, &nX, 0); cBYE(status);
  ptr_w->wq = (uint8_t *)X; X = NULL;
  ptr_w->sz_wq = nX; nX = 0;
  //-------------------------------------------------------
  // create wk
  status = rs_mmap("_wk.bin", &X, &nX, 0); cBYE(status);
  ptr_w->wk = (uint8_t *)X; X = NULL;
  ptr_w->sz_wk = nX; nX = 0;
  //-------------------------------------------------------
  // create wv
  status = rs_mmap("_wv.bin", &X, &nX, 0); cBYE(status);
  ptr_w->wv = (uint8_t *)X; X = NULL;
  ptr_w->sz_wv = nX; nX = 0;
  //-------------------------------------------------------
  // create wo
  status = rs_mmap("_wo.bin", &X, &nX, 0); cBYE(status);
  ptr_w->wo = (uint8_t *)X; X = NULL;
  ptr_w->sz_wo = nX; nX = 0;
  //-------------------------------------------------------
  // create rms_ffn_weight
  status = rs_mmap("_rms_ffn_weight.bin", &X, &nX, 0); cBYE(status);
  ptr_w->rms_ffn_weight = (uint8_t *)X; X = NULL;
  ptr_w->sz_rms_ffn = nX; nX = 0;
  //-------------------------------------------------------
  // create w1
  status = rs_mmap("_w1.bin", &X, &nX, 0); cBYE(status);
  ptr_w->w1 = (uint8_t *)X; X = NULL;
  ptr_w->sz_w1 = nX; nX = 0;
  //-------------------------------------------------------
  // create w2
  status = rs_mmap("_w2.bin", &X, &nX, 0); cBYE(status);
  ptr_w->w2 = (uint8_t *)X; X = NULL;
  ptr_w->sz_w2 = nX; nX = 0;
  //-------------------------------------------------------
  // create w3
  status = rs_mmap("_w3.bin", &X, &nX, 0); cBYE(status);
  ptr_w->w3 = (uint8_t *)X; X = NULL;
  ptr_w->sz_w3 = nX; nX = 0;
  //-------------------------------------------------------
  // create rms_final_weight
  status = rs_mmap("_rms_final_weight.bin", &X, &nX, 0); cBYE(status);
  ptr_w->rms_final_weight = (uint8_t *)X; 
  ptr_w->sz_rms_final = nX; 
  //-------------------------------------------------------
  status = rs_mmap("_wcls.bin", &X, &nX, 0); cBYE(status);
  ptr_w->wcls = ptr_w->token_embedding_table; // TODO P1 HACK 
  ptr_w->sz_wcls = ptr_w->sz_tet;
  //-------------------------------------------------------

BYE:
  return status;
}

int
qnt_munmap_weights(
    TransformerWeights * restrict ptr_w
    )
{
  int status = 0;

  // TODO 
  mcr_rs_munmap(ptr_w->token_embedding_table, ptr_w->sz_tet);
  mcr_rs_munmap(ptr_w->rms_att_weight, ptr_w->sz_rms_att);
  mcr_rs_munmap(ptr_w->wq, ptr_w->sz_wq);
  mcr_rs_munmap(ptr_w->wk, ptr_w->sz_wk);
  mcr_rs_munmap(ptr_w->wv, ptr_w->sz_wv);
  mcr_rs_munmap(ptr_w->wo, ptr_w->sz_wo);
  mcr_rs_munmap(ptr_w->rms_ffn_weight, ptr_w->sz_rms_ffn);
  mcr_rs_munmap(ptr_w->w1, ptr_w->sz_w1);
  mcr_rs_munmap(ptr_w->w2, ptr_w->sz_w2);
  mcr_rs_munmap(ptr_w->w3, ptr_w->sz_w3);
  mcr_rs_munmap(ptr_w->rms_final_weight, ptr_w->sz_rms_final);
  mcr_rs_munmap(ptr_w->wcls, ptr_w->sz_wcls);

  return status;
}

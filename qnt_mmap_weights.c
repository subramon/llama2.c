#include <stdio.h>
#include "macros.h"
#include "rs_mmap.h"
#include "consts.h"
#include "qnt_weights_file_layout.h"
#include "qnt_mmap_weights.h"

int
qnt_mmap_weights(
    QntTransformerWeights * restrict ptr_w
    )
{
  int status = 0;
  char *X = NULL; size_t nX = 0;
  // create token_embedding_table
  status = rs_mmap("_token_embedding_table.ui8", &X, &nX, 0); cBYE(status);
  ptr_w->qnt_tet = (uint8_t *)X; X = NULL;
  ptr_w->sz_qnt_tet = nX; nX = 0;
  //-------------------------------------------------------
  // create rms_att_weight
  status = rs_mmap("_rms_att_weight.ui8", &X, &nX, 0); cBYE(status);
  ptr_w->qnt_rms_att_weight = (uint8_t *)X; X = NULL;
  ptr_w->sz_qnt_rms_att = nX; nX = 0;
  //-------------------------------------------------------
  // create wq
  status = rs_mmap("_wq.ui8", &X, &nX, 0); cBYE(status);
  ptr_w->qnt_wq = (uint8_t *)X; X = NULL;
  ptr_w->sz_qnt_wq = nX; nX = 0;
  //-------------------------------------------------------
  // create wk
  status = rs_mmap("_wk.ui8", &X, &nX, 0); cBYE(status);
  ptr_w->qnt_wk = (uint8_t *)X; X = NULL;
  ptr_w->sz_qnt_wk = nX; nX = 0;
  //-------------------------------------------------------
  // create wv
  status = rs_mmap("_wv.ui8", &X, &nX, 0); cBYE(status);
  ptr_w->qnt_wv = (uint8_t *)X; X = NULL;
  ptr_w->sz_qnt_wv = nX; nX = 0;
  //-------------------------------------------------------
  // create wo
  status = rs_mmap("_wo.ui8", &X, &nX, 0); cBYE(status);
  ptr_w->qnt_wo = (uint8_t *)X; X = NULL;
  ptr_w->sz_qnt_wo = nX; nX = 0;
  //-------------------------------------------------------
  // create rms_ffn_weight
  status = rs_mmap("_rms_ffn_weight.ui8", &X, &nX, 0); cBYE(status);
  ptr_w->qnt_rms_ffn_weight = (uint8_t *)X; X = NULL;
  ptr_w->sz_qnt_rms_ffn = nX; nX = 0;
  //-------------------------------------------------------
  // create w1
  status = rs_mmap("_w1.ui8", &X, &nX, 0); cBYE(status);
  ptr_w->qnt_w1 = (uint8_t *)X; X = NULL;
  ptr_w->sz_qnt_w1 = nX; nX = 0;
  //-------------------------------------------------------
  // create w2
  status = rs_mmap("_w2.ui8", &X, &nX, 0); cBYE(status);
  ptr_w->qnt_w2 = (uint8_t *)X; X = NULL;
  ptr_w->sz_qnt_w2 = nX; nX = 0;
  //-------------------------------------------------------
  // create w3
  status = rs_mmap("_w3.ui8", &X, &nX, 0); cBYE(status);
  ptr_w->qnt_w3 = (uint8_t *)X; X = NULL;
  ptr_w->sz_qnt_w3 = nX; nX = 0;
  //-------------------------------------------------------
  // create rms_final_weight
  status = rs_mmap("_rms_final_weight.ui8", &X, &nX, 0); cBYE(status);
  ptr_w->qnt_rms_final_weight = (uint8_t *)X; 
  ptr_w->sz_qnt_rms_final = nX; 
  //-------------------------------------------------------
  status = rs_mmap("_wcls.ui8", &X, &nX, 0); cBYE(status);
  ptr_w->qnt_wcls = ptr_w->qnt_tet; // TODO P1 HACK 
  ptr_w->sz_qnt_wcls = ptr_w->sz_qnt_tet; // TODO P1 HACK
  //-------------------------------------------------------

BYE:
  return status;
}

int
qnt_munmap_weights(
    QntTransformerWeights * restrict ptr_w
    )
{
  int status = 0;

  mcr_rs_munmap(ptr_w->qnt_tet, ptr_w->sz_qnt_tet);
  mcr_rs_munmap(ptr_w->qnt_rms_att_weight, ptr_w->sz_qnt_rms_att);
  mcr_rs_munmap(ptr_w->qnt_wq, ptr_w->sz_qnt_wq);
  mcr_rs_munmap(ptr_w->qnt_wk, ptr_w->sz_qnt_wk);
  mcr_rs_munmap(ptr_w->qnt_wv, ptr_w->sz_qnt_wv);
  mcr_rs_munmap(ptr_w->qnt_wo, ptr_w->sz_qnt_wo);
  mcr_rs_munmap(ptr_w->qnt_rms_ffn_weight, ptr_w->sz_qnt_rms_ffn);
  mcr_rs_munmap(ptr_w->qnt_w1, ptr_w->sz_qnt_w1);
  mcr_rs_munmap(ptr_w->qnt_w2, ptr_w->sz_qnt_w2);
  mcr_rs_munmap(ptr_w->qnt_w3, ptr_w->sz_qnt_w3);
  mcr_rs_munmap(ptr_w->qnt_rms_final_weight, ptr_w->sz_qnt_rms_final);
  mcr_rs_munmap(ptr_w->qnt_wcls, ptr_w->sz_qnt_wcls);

  return status;
}

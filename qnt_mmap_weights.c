#include <stdio.h>
#include "macros.h"
#include "rs_mmap.h"
#include "consts.h"
#include "qnt_weights_file_layout.h"
#include "qnt_mmap_weights.h"

static int
foo(
    const char * const prefix,
    uint8_t **ptr_qnt,
    size_t *ptr_sz_qnt,
    float **ptr_off,
    size_t *ptr_sz_off,
    float **ptr_del,
    size_t *ptr_sz_del
   )
{
  int status = 0;
  char *X = NULL; size_t nX = 0;
  size_t len = strlen(prefix) + 16;
  char *file_name = malloc(len);

  sprintf(file_name, "%s.ui8", prefix);
  status = rs_mmap(file_name, &X, &nX, 0); cBYE(status);
  *ptr_qnt = (uint8_t *)X; X = NULL;
  *ptr_sz_qnt =  nX; nX = 0;

  sprintf(file_name, "%s.offset", prefix);
  status = rs_mmap(file_name, &X, &nX, 0); cBYE(status);
  *ptr_off = (float *)X; X = NULL;
  *ptr_sz_off =  nX; nX = 0;

  sprintf(file_name, "%s.delta", prefix);
  status = rs_mmap(file_name, &X, &nX, 0); cBYE(status);
  *ptr_del = (float *)X; X = NULL;
  *ptr_sz_del =  nX; nX = 0;

BYE:
  free_if_non_null(file_name);
  return status;
}

int
qnt_mmap_weights(
    QntTransformerWeights * restrict ptr_w
    )
{
  int status = 0;
  char *X = NULL; size_t nX = 0;
  // create token_embedding_table
  status = foo("_token_embedding_table",
  &(ptr_w->qnt_tet), &(ptr_w->sz_qnt_tet), 
  &(ptr_w->offset_tet), &(ptr_w->sz_offset_tet), 
  &(ptr_w->delta_tet), &(ptr_w->sz_delta_tet));
  //-------------------------------------------------------
  // create rms_att_weight
  status = foo("_rms_att_weight",
  &(ptr_w->qnt_rms_att_weight), &(ptr_w->sz_qnt_rms_att_weight), 
  &(ptr_w->offset_rms_att), &(ptr_w->sz_offset_rms_att), 
  &(ptr_w->delta_rms_att), &(ptr_w->sz_delta_rms_att));
  //-------------------------------------------------------
  // create wq
  status = foo("_wq",
  &(ptr_w->qnt_wq), &(ptr_w->sz_qnt_wq), 
  &(ptr_w->offset_wq), &(ptr_w->sz_offset_wq), 
  &(ptr_w->delta_wq), &(ptr_w->sz_delta_wq));
  //-------------------------------------------------------
  // create wk
  status = foo("_wk",
  &(ptr_w->qnt_wk), &(ptr_w->sz_qnt_wk), 
  &(ptr_w->offset_wk), &(ptr_w->sz_offset_wk), 
  &(ptr_w->delta_wk), &(ptr_w->sz_delta_wk));
  //-------------------------------------------------------
  // create wv
  status = foo("_wv",
  &(ptr_w->qnt_wv), &(ptr_w->sz_qnt_wv), 
  &(ptr_w->offset_wv), &(ptr_w->sz_offset_wv), 
  &(ptr_w->delta_wv), &(ptr_w->sz_delta_wv));
  //-------------------------------------------------------
  // create wo
  status = foo("_wo",
  &(ptr_w->qnt_wo), &(ptr_w->sz_qnt_wo), 
  &(ptr_w->offset_wo), &(ptr_w->sz_offset_wo), 
  &(ptr_w->delta_wo), &(ptr_w->sz_delta_wo));
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
#ifdef TODO 
  // create rms_final_weight
  status = rs_mmap("_rms_final_weight.ui8", &X, &nX, 0); cBYE(status);
  ptr_w->qnt_rms_final_weight = (uint8_t *)X; 
  ptr_w->sz_qnt_rms_final = nX; 
  //-------------------------------------------------------
  status = rs_mmap("_wcls.ui8", &X, &nX, 0); cBYE(status);
  ptr_w->qnt_wcls = ptr_w->qnt_tet; // TODO P1 HACK 
  ptr_w->sz_qnt_wcls = ptr_w->sz_qnt_tet; // TODO P1 HACK
  //-------------------------------------------------------
#endif
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
  mcr_rs_munmap(ptr_w->qnt_rms_att_weight, ptr_w->sz_qnt_rms_att_weight);
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

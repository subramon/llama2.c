#include <stdio.h>
#include "macros.h"
#include "consts.h"
#include "rs_mmap.h"
#include "qntz_2d.h"

int
qntz_2d(
    const char * const orig_file,
    const char * const qnt_file,
    const char * const off_file,
    const char * const del_file,
    int nR,
    int nC
    )
{
  int status = 0;
  FILE *ofp = NULL;
  FILE *dfp = NULL;
  FILE *qfp = NULL;
  char *X = NULL; size_t nX = 0;
  size_t ispc_nC = mcr_round_up(nC);
  // create token_embedding_table
  status = rs_mmap(orig_file, &X, &nX, 0); cBYE(status);
  float *fX = (float *)X;
  if ( (size_t)(nR * nC) * sizeof(float) != nX ) { go_BYE(-1); }

  qfp = fopen(qnt_file, "wb");
  return_if_fopen_failed(qfp, qnt_file, "wb");
  ofp = fopen(off_file, "wb");
  return_if_fopen_failed(ofp, off_file, "wb");
  dfp = fopen(del_file, "wb");
  return_if_fopen_failed(dfp, del_file, "wb");
  size_t idx = 0;
  for ( int i = 0; i < nR; i++ ) { 
    // save current location because we need to scan row again
    size_t bak_idx = idx;
    // scan the row to get min/max values  and compute delta/offset
    float minval = fX[idx];
    float maxval = fX[idx];
    for ( int j = 0; j < nC; j++ ) { 
      if ( fX[idx] < minval ) { minval = fX[idx]; }
      if ( fX[idx] > maxval ) { maxval = fX[idx]; }
      idx++;
    }
    float offset = minval;
    float delta = (maxval - minval)/255; // TODO 255 or 256?
    fwrite(&offset, sizeof(float), 1, ofp);
    fwrite(&delta,  sizeof(float), 1, dfp);
    //------------------------------------
    // Start scanning line again and quantizing as you go 
    idx = bak_idx;
    int j = 0;
    for ( ; j < nC; j++ ) { 
      uint8_t qval = (uint8_t)((fX[idx] - minval) * delta);
      fwrite(&qval,  sizeof(uint8_t), 1, qfp);
      idx++;
    }
    // Pad for memory alignment 
    for ( ; j < (int)ispc_nC; j++ ) { 
      uint8_t qval = 0;
      fwrite(&qval,  sizeof(uint8_t), 1, qfp);
    }
  }
BYE:
  fclose_if_non_null(qfp);
  fclose_if_non_null(dfp);
  fclose_if_non_null(ofp);
  return status;
}

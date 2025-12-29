#ifndef __Q_MACROS_H
#define __Q_MACROS_H
#define WHEREAMI { fprintf(stderr, "Line %3d of File %s \n", __LINE__, __FILE__);  }
/*-------------------------------------------------------*/
#define go_BYE(x) { WHEREAMI; status = x ; goto BYE; }
/*-------------------------------------------------------*/

#define cBYE(x) { if ( (x) != 0 ) { go_BYE((x)) } }
#define fclose_if_non_null(x) { if ( (x) != NULL ) { fclose((x)); (x) = NULL; } } 
#define free_if_non_null(x) { if ( (x) != NULL ) { free((x)); (x) = NULL; } }
#define return_if_fopen_failed(fp, file_name, access_mode) { if ( fp == NULL ) { fprintf(stderr, "Unable to open file %s for %s \n", file_name, access_mode); go_BYE(-1); } }
#define return_if_malloc_failed(x) { if ( x == NULL ) { fprintf(stderr, "Unable to allocate memory\n"); go_BYE(-1); } }
#define return_if_null_str(x) { if ( ( x == NULL ) ||( *x == '\0' ) ) { \
  go_BYE(-1); } \
}

#define mcr_nop(X)  ((X))
#define mcr_sqr(X)  ((X) * (X))
#define mcr_min(X, Y)  ((X) < (Y) ? (X) : (Y))
#define mcr_add(X, Y)  ((X) + (Y) )
#define mcr_max(X, Y)  ((X) > (Y) ? (X) : (Y))
#define mcr_sum(X, Y)  ((X) + (Y))
#define mcr_sum_sqr(X, Y)  ((X) + (Y)*(Y))
#define sqr(X)  ((X) * (X))

#define mcr_rs_munmap(X, nX) { \
  if ( ( X == NULL ) && ( nX != 0 ) ) {  WHEREAMI; } \
  if ( ( X != NULL ) && ( nX == 0 ) )  { WHEREAMI; } \
  if ( X != NULL ) { \
  	int l_status = munmap(X, nX); if ( l_status != 0 ) { WHEREAMI; } \
  } \
  X = NULL; nX = 0;  \
}

#define unlink_if_non_null(x) { if ( x != NULL ) { unlink( x ); } }

#define mcr_alloc_null_str(x, y) { \
  x = (char *)malloc(y * sizeof(char)); \
  return_if_malloc_failed(x); \
  zero_string(x, y); \
}

#define asm_time(x) { \
  __asm__{  \
    RDTSC  \
      mov DWORD PTR x, eax \
      mov DWORD PTR x+4, eax \
  } \
}

#define mcr_get_2d_ptr(P, x, nX) { (P + (x*nX)) }
#define mcr_get_3d_ptr(P, x, y, nX, nY) { (P + (x*nX*nY) + (y*nY)) }
#define mcr_round_up(x) { (uint32_t)(x) }

#endif

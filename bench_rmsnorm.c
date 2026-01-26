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
#include "macros.h"
#include "rmsnorm.h"

int
main(
    int argc,
    char **argv
    )
{
  int status = 0;
  int n = 1024; 
  srand48(time(NULL));
  for ( register int i = 0; i < 10; i++ ) { 
    float *o = NULL, *x = NULL, *w = NULL;

    status = posix_memalign(&o, 32, (n * sizeof(float)));
    status = posix_memalign(&w, 32, (n * sizeof(float)));
    status = posix_memalign(&x, 32, (n * sizeof(float)));
    uint64_t t = 0;
    int iters = 100;
    for ( register int j = 0; j < iters; j++ ) {
      // initialize
      for ( int k = 0; k < n; k++ ) { 
        o[k] = (float)drand48();
        x[k] = (float)drand48();
        w[k] = (float)drand48();
      }
      uint64_t t0 = __rdtsc();
      rmsnorm(o, x, w, n); 
      t += (__rdtsc() - t0);
    }
    printf("%8d %lf \n", n, ((double)t/(double)iters));
    free(o);
    free(x);
    free(w);
    n *= 2;
  }
  return status;
}

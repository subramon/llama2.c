#ifndef __CONSTS_H
#define __CONSTS_H
#define BYTES_IN_REG  16 // TODO P0 32 for AVX2
#define FLOATS_IN_REG ((unsigned int)4) // TODO P0. 8 for AVX2
#define CACHE_LINE ((unsigned int)64 )
#define FLOATS_IN_CACHE_LINE (CACHE_LINE / sizeof(float))
#endif // __CONSTS_H

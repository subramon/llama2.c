#ifndef __CONSTS_H
#define __CONSTS_H
#define BYTES_IN_REG  32
#define FLOATS_IN_REG ((unsigned int)8)
#define CACHE_LINE ((unsigned int)64 )
#define FLOATS_IN_CACHE_LINE (CACHE_LINE / sizeof(float))
#endif // __CONSTS_H

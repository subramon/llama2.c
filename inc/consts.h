#ifndef __CONSTS_H
#define __CONSTS_H
#define FLOATS_IN_REG ((unsigned int)4)
#define LOG2_FLOATS_IN_REG ((unsigned int)2 )
#define CACHE_LINE ((unsigned int)64 )
#define FLOATS_IN_CACHE_LINE (CACHE_LINE / sizeof(float))
#endif // __CONSTS_H

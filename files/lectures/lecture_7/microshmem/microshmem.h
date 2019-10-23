#ifndef MICROSHMEM_H
#define MICROSHMEM_H

    extern long PORTIONSIZE, PORTIONINCREMENT;

#ifdef __cplusplus
extern "C" {
#endif

    extern void shmem_long_put( long *dstaddr, long *srcaddr, long len, int node );
    extern void shmem_long_p( long *dstaddr, long src, int node );
    extern void shmem_barrier_all( void );
    extern void shmem_long_get( long *dstaddr, long *srcaddr, long lng, int pe );
    extern void *shmalloc( long lng );
    extern void shmem_init( int*, char *** );
    extern void shmem_finalize( void );
    extern void touch_long( long val );
    extern void touch_double( double val );
    extern int shmem_my_pe( void );
    extern int shmem_n_pes( void );
    extern double shmem_time( void );
    extern void shmem_coarray_all( void *section, long sectionsize, void *capointer[] );

#ifdef __cplusplus
}
#endif

#define shmem_double_put( dst, src, len, node ) shmem_long_put( ((long*)(dst)), ((long*)(src)), ((len)*(sizeof(double)/sizeof(long))), (node) )
#define shmem_double_p( dst, src, node ) shmem_long_put( ((long*)(dst)), ((long*)(&(src))), (sizeof(double)/sizeof(long)), (node) )
// NB!!! shmem_put transfers only len == multiple of sizeof(long) !!!!
#define shmem_put( dst, src, len, node ) shmem_long_put( ((long*)(dst)), ((long*)(src)), (((len)+sizeof(long)-1)/sizeof(long)), (node) ) 

#endif

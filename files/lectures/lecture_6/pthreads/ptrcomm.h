#include <stdio.h>
    extern volatile int ptr_done[];
    extern volatile int ptr_barr[];
    extern void ptr_launch( void *(*main_f)(void*), int nthreads );
#define MYTHREAD( ptrarg ) (((long)(ptrarg)) & 0xffff)
#define NTHREADS( ptrarg ) (((long)(ptrarg)) >> 16)
#define MARK_OVER( mythread ) ptr_done[mythread] = 1
    extern void ptr_barrier( int mythread );
    extern void ptr_send_r( int mythread, int to, void *array );
    extern void ptr_send_w( int mythread, int to );
    extern void ptr_recv_w( int mythread, int from, void *array, int leng );

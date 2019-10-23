#include <stdio.h>
#include "ptrcomm.h"
    void* ptr_main( void *parg )
{
    int i, mythread, nthreads;
//
    mythread = MYTHREAD( parg );
    nthreads = NTHREADS( parg );
    printf( "I am %d of %d, starting\n", mythread, nthreads );
//    sleep( 10 );
    for ( i = 0; i < 1000; i++ ) ptr_barrier( mythread );
    printf( "I am %d of %d, finishing\n", mythread, nthreads );
    MARK_OVER( mythread );
    return 0;    
}    
    int main( int argc, char **argv )
{
    int i, total;
//
    ptr_launch( ptr_main, 10 );
    return 0;
}

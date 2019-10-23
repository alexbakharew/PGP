#include <stdio.h>
#include <malloc.h>
#include "ptrcomm.h"
#define SIZMES 20000
    void* ptr_main( void *parg )
{
    int i, j, k, times, mythread, nthreads;
    int *buffers[100];
//
    mythread = MYTHREAD( parg );
    nthreads = NTHREADS( parg );
    for ( i = 0; i < nthreads; i++ )
     {
      if ( (buffers[i] = malloc( SIZMES*sizeof( int ) )) == NULL )
       {
        fprintf( stderr, "No memory\n" );
        return( 0 );
       }
     }
    printf( "I am %d of %d, starting\n", mythread, nthreads );
    times = 10;
    printf( "Running all routes %d times\n", times );
    for ( k = 0; k < SIZMES; k++ ) (buffers[mythread])[k] = mythread + k;
    while ( times-- )
     {
      for ( i = 0; i < nthreads; i++ )
       {
        if ( i != mythread )
         {
          for ( k = 0; k < SIZMES; k++ ) (buffers[i])[k] = -1;
         }
       }
      j = 0;
      for ( i = 0; i < nthreads; i++ )
       {
        if ( i != mythread )
         {
          ptr_send_r( mythread, i, (void*)buffers[mythread] );
         }
       }
      for ( i = 0; i < nthreads; i++ )
       {
        if ( i != mythread )
         {
          ptr_recv_w( mythread, i, buffers[i], SIZMES*sizeof( int ) );
          for ( k = 0; k < SIZMES; k++ )
           {
            if ( (buffers[i])[k] != (i+k) )
             {
              printf( "Processor %d received %d instead of %d from %d\n",
                           mythread, (buffers[i])[k], i+k, i ); 
              j++;
             }
           }
         }
       }
      for ( i = 0; i < nthreads; i++ )
       {
        if ( i != mythread )
         {
          ptr_send_w( mythread, i );
         }
       }
      printf( "Processor %d received from all with %d errors\n", mythread, j );
      fflush( stdout );
     }  
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#define MAX_TO_LAUNCH 100
#define FROM_TO( i, j ) ((i)*MAX_TO_LAUNCH+(j))
    static pthread_t ptr_id[MAX_TO_LAUNCH];
    volatile int ptr_done[MAX_TO_LAUNCH];
    volatile int ptr_barr[MAX_TO_LAUNCH];
    volatile void *ptr_mesg[MAX_TO_LAUNCH*MAX_TO_LAUNCH];
    static int ptr_num = 0;
    void ptr_launch( void *(*main_f)(void*), int nthreads )
{
    int mythread, j;
    long ptrarg;
//
    for ( mythread = 0; mythread < nthreads; mythread++ )
     {
      if ( ptr_num >= MAX_TO_LAUNCH )
       {
        fprintf( stderr, "Too many threads to launch\n" );
       }
      ptrarg = (nthreads << 16) | mythread;
      ptr_done[ptr_num] = 0;
      ptr_barr[ptr_num] = 0;
      for ( j = 0; j < nthreads; j++ ) ptr_mesg[FROM_TO(ptr_num,j)] = 0;
      pthread_create( ptr_id+ptr_num, 0, main_f, (void*)ptrarg );
      ptr_num++;
     }
    while ( 1 )
     {
      sleep( 1 );
      ptrarg = 0;
      for ( mythread = 0; mythread < ptr_num; mythread++ ) ptrarg += ptr_done[mythread];
      if ( ptrarg >= ptr_num )
       {
        printf( "All threads are over, terminating main process\n" );
        exit( 0 ); 
       }
     }   
}
    void ptr_barrier( int mythread )
{
    int i, nb;
//    
    if ( mythread == 0 )
     {
      ptr_barr[0] = 1;
      while ( 1 )
       {
	sched_yield();
        nb = 0;
        for ( i = 0; i < ptr_num; i++ ) nb += ptr_barr[i];
        if ( nb >= ptr_num ) break;
       }
      for ( i = 0; i < ptr_num; i++ ) ptr_barr[i] = 0;
     }
    else
     {
      ptr_barr[mythread] = 1;
      while ( ptr_barr[mythread] == 1 ) sched_yield(); 
     }   
}
    void ptr_send_r( int mythread, int to, void *array )
{     
    ptr_mesg[FROM_TO(mythread,to)] = array;
}
    void ptr_send_w( int mythread, int to )
{     
    while( ptr_mesg[FROM_TO(mythread,to)] ) sched_yield();
}
    void ptr_recv_w( int mythread, int from, void *array, int leng )
{
    while ( ptr_mesg[FROM_TO(from,mythread)] == 0 ) sched_yield();
    memcpy( array, (void*)(ptr_mesg[FROM_TO(from,mythread)]), leng );
    ptr_mesg[FROM_TO(from,mythread)] = 0;
}

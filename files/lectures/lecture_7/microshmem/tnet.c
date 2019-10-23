#include <stdio.h>
#include <sys/time.h>
#include "microshmem.h"
#define LENGTH 10000
#define TIMES 10000000
#define STEPITER 10
	int main( int argc, char **argv )
{
	int i, j, k, n, mynode, nodes, localsize, nerr, totalerr, t;
	static long local[10000000];
	static long counter[10000000];
	long barrier;
	static int portion[1000]; // be very optimistic
/***/
/*
	PORTIONSIZE = 100;
	PORTIONINCREMENT = 50;
*/
	MPI_Init( &argc, &argv );
	mynode = shmem_my_pe();
	nodes = shmem_n_pes();
	for ( k = 0; k < nodes; k++ )
	 {
	  shmem_barrier_all(); 
	  if (k == mynode)
	   {
	    printf( "I am %d of %d\n", mynode, nodes );
	    for ( j = 0; j < LENGTH; j++ ) counter[j] = j + mynode;
	    for ( i = 0; i < nodes; i++ ) portion[i] = i*LENGTH;
	    n = TIMES;
	    printf( "Tnet-ing shmem of %d words %d times\n", LENGTH, n );
	   }
	 }      
	totalerr = 0;
	for ( i = 0; i < nodes; i++ )
	 {
	  for ( j = 0; j < LENGTH; j++ )
	   {
	    local[portion[i]+j] = -1;
	   }
	 }  
	while ( n-- )
	 {
	  shmem_barrier_all();
	  for ( i = 0; i < nodes; i++ )
	   {
	    if ( i != mynode ) shmem_long_put( local+portion[mynode], counter, LENGTH, i );
	   }
	  shmem_barrier_all(); 
	  nerr = 0;
	  for ( i = 0; i < nodes; i++ )
	   {
	    if ( i != mynode )
	     {
	      for ( j = 0; j < LENGTH; j++ )
	       {
	        if ( local[portion[i]+j] != (j + i) )
	         {
	          printf( "Processor %d received %ld instead of %d from %d\n", mynode, local[portion[i]+j], i + j, i );
		  nerr++;
	         }
	        local[portion[i]+j] = -1;
	       }
	     }    
	   } 
	  printf( "Processor %d received from all with %d errors \n", mynode, nerr ); 
	  totalerr += nerr;
	  if ( !(n%STEPITER) ) printf( "Total errors: %d\n", totalerr );
	 }
	MPI_Finalize();
        return 0;
}

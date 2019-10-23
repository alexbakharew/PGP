#include <upc_io.h>
#include <upc_relaxed.h>
#include <stdio.h>
#define MX 640
#define MY 480
#define NITER 200
#define STEPITER 100
#define BLOCK (((MX+THREADS-1)/THREADS)*MY)
      shared [BLOCK] double sfold[BLOCK*THREADS];
      shared [BLOCK] double sfnew[BLOCK*THREADS];
      int main( int argc, char **argv )
{
      int i, j, n, nrows;
      double *plower, *pold, *pupper, *pnew;
      shared [BLOCK] double *fold;
      shared [BLOCK] double *fnew;
      shared [BLOCK] double *ptmp;
      static double rowbuf[MY];
      upc_file_t *fp;
      upc_hint_t hint;
//
      for ( j = 0; j < MY; j++ ) rowbuf[j] = 0.0;
      printf( "Solving heat conduction task on %d by %d grid\n", MX, MY ); 
      fflush( stdout );
      fold = sfold;
      fnew = sfnew;
/* Initial conditions: */
      nrows = 0;
      upc_forall ( i = 0; i < MX; i++; &(fold[i*MY]) )
       {
        nrows++;
	pold = (double*)(&(fold[i*MY]));	
	pnew = (double*)(&(fnew[i*MY]));
        for ( j = 0; j < MY; j++ )
         {
           pold[j] = pnew[j] = 0.0;
           if      ( (i == 0)
                  || (j == 0) ) pold[j] = pnew[j] = 1.0;
           else if ( (i == (MX-1))
                  || (j == (MY-1)) ) pold[j] = pnew[j] = 0.5;
         }
       }
// Iteration loop: 
      for ( n = 0; n < NITER; n++ )
       {
        if ( !(n%STEPITER) ) printf( "Iteration %d\n", n );
// Step of calculation starts here: 
	upc_barrier;
        upc_forall ( i = 1; i < (MX-1); i++; &(fold[i*MY]) )
         {
	  pold = (double*)(&(fold[i*MY]));	
	  pnew = (double*)(&(fnew[i*MY]));
	  if ( upc_threadof( &(fold[(i-1)*MY]) ) != MYTHREAD ) 
	   {
	    upc_memget( rowbuf, &(fold[(i-1)*MY]), MY*sizeof( fold[0] ) );
	    plower = rowbuf;
	   }
	  else
	   {
	    plower = (double*)(&(fold[(i-1)*MY]));
	   }    
	  if ( upc_threadof( &(fold[(i+1)*MY]) ) != MYTHREAD )
	   {
	    upc_memget( rowbuf, &(fold[(i+1)*MY]), MY*sizeof( fold[0] ) );
	    pupper = rowbuf;
	   }
	  else  
	   pupper = (double*)(&(fold[(i+1)*MY]));
          for ( j = 1; j < (MY-1); j++ )
           {
            pnew[j] = ( pold[j+1] + pold[j-1] + plower[j] + pupper[j] )*0.25;
           }
         }
	upc_barrier; 
// Exchange old/new pointers: 
	ptmp = fold;
        fold = fnew;
        fnew = ptmp;
       }
/* Calculation is done, fold array is a result: */ 
      fp = upc_all_fopen( "progrev.dat", 
    			    UPC_COMMON_FP |
			    UPC_WRONLY |
			    UPC_CREATE |
			    UPC_TRUNC, 0, &hint );
      if ( !fp )
       {
        fprintf( stderr, "Node %d failed to open output file\n", MYTHREAD );
	return -1;
       }
      upc_all_fwrite_shared( fp, &fold[0], nrows*MY, sizeof( fold[0] ), MX*MY, UPC_IN_ALLSYNC | UPC_OUT_ALLSYNC );
      upc_all_fclose( fp );      
      return 0;
}

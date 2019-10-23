// Shmem subset trivial implementation.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
// User-tunable constants:
    long PORTIONSIZE      = 1000000l;
    long PORTIONINCREMENT = 1000000l;
//
    MPI_Comm shmemcomm;
    static int mynode, nodes;
// Send and recv arrays:
    static long *sendarray, *recvarray;
// Lengthes, displacements:
    static int *sendcount, *senddispl, *recvcount, *recvdispl;
// Function to provide extra place (at least len longs) in a buffer for a given node:
    static void provide( long **array, int *displ, int *count, int node, long len )
{
    long *parray;
    int i;
//
    if ( len > 0 )
     {
      len += (PORTIONINCREMENT-1); len /= PORTIONINCREMENT; len *= PORTIONINCREMENT;
      parray = malloc( (displ[nodes]+len)*sizeof(*parray) );
      if ( !parray )
       {
        fprintf( stderr, "Microshmem: no memory in realloc\n" );
        exit( -1 );
       } 
      for ( i = 0; i <= node; i++ )
       {  
        if ( count[i] ) memcpy( parray+displ[i], (*array)+displ[i], count[i]*sizeof(*parray) );
       }
      for ( i = node+1; i < nodes; i++ )
       {
        if ( count[i] ) memcpy( parray+displ[i]+len, (*array)+displ[i], count[i]*sizeof(*parray) );
        displ[i] += len;
       } 
      displ[nodes] += len; 
      free( *array );
      *array = parray;
     }
}
// Generic send function:
    void shmem_long_put( long *dstaddr, long *srcaddr, long len, int node )
{
    long nfree;
    long *pnode;
//
    if      ( (node < 0) || (node >= nodes) || (len <= 0) ) return;
    if ( node == mynode )
     memcpy( dstaddr, srcaddr, len*sizeof(*dstaddr) );
    else
     { 
      nfree = senddispl[node+1] - (senddispl[node]+sendcount[node]);
      provide( &sendarray, senddispl, sendcount, node, (len+2)-nfree );
      pnode = sendarray+senddispl[node]+sendcount[node];
      *pnode++ = len;
      *pnode++ = (long)dstaddr;
      memcpy( pnode, srcaddr, len*sizeof(*pnode) );
      sendcount[node] += (len+2);
     } 
}
//
    void shmem_long_p( long *dstaddr, long src, int node )
{
    shmem_long_put( dstaddr, &src, 1, node );
}
//
    void shmem_barrier_all( void )
{
    int i, j, n, lng;
    long *parray, *dstaddr;
// Exchange the send length data:
    MPI_Alltoall( sendcount, 1, MPI_INT, recvcount, 1, MPI_INT, shmemcomm ); 
// Provide enough place for recv:
    lng = recvdispl[nodes];
    recvdispl[0] = 0;
    for ( i = 0; i < nodes; i++ )
     {
      recvdispl[i+1] = recvdispl[i] + recvcount[i];
     } 
    if ( recvdispl[nodes] > lng )
     {
// Realloc is necessary:
      free( recvarray );
      recvarray = malloc( recvdispl[nodes]*sizeof(*recvarray) );
      if ( !recvarray )
       {
        fprintf( stderr, "Microshmem: no memory in realloc\n" );
        exit( -1 );
       } 
     } 
    else
     {
// Restore the length of the whole array to avoid unnecessary reallocs in future:
      recvdispl[nodes] = lng; 
     } 
// Exchange the data itself:
    MPI_Alltoallv( sendarray, sendcount, senddispl, MPI_LONG, recvarray, recvcount, recvdispl, MPI_LONG, shmemcomm );
// Process the received data:
    for ( i = 0; i < nodes; i++ )
     {
      parray = recvarray + recvdispl[i];
      n = recvcount[i];
      j = 0;
      while ( j < n )
       {
        lng = parray[j++];
        dstaddr = (long*)(parray[j++]);
        memcpy( dstaddr, parray+j, lng*sizeof(*parray) );
        j += lng;
       }
      sendcount[i] = recvcount[i] = 0; 
     }
}
    void shmem_long_get( long *dstaddr, long *srcaddr, long lng, int pe )
{
// Not implemented, but should be present:
}
    void *shmalloc( long lng )
{
// Not implemented, but should be present:
    return 0;
}
    void shmem_init( int *argc, char ***argv )
{
    MPI_Init( argc, argv );
}
    void shmem_finalize( void )
{
    MPI_Finalize();
}
    void touch_long( long val )
{
}
    void touch_double( double val )
{
}
    int shmem_my_pe( void )
{
    return mynode;
}
    int shmem_n_pes( void )
{
    return nodes;
}
    int MPI_Init( int *argc, char ***argv )
{
    int i, rc;
//
    rc = PMPI_Init( argc, argv );
    if ( rc != MPI_SUCCESS ) return rc;
    MPI_Comm_dup( MPI_COMM_WORLD, &shmemcomm ); 
    MPI_Comm_size( shmemcomm, &nodes );
    MPI_Comm_rank( shmemcomm, &mynode );
    sendarray = malloc( nodes*PORTIONSIZE*sizeof(*sendarray) );
    recvarray = malloc( nodes*PORTIONSIZE*sizeof(*recvarray) );
    sendcount = malloc( nodes*sizeof(*sendcount) );
    recvcount = malloc( nodes*sizeof(*recvcount) );
    senddispl = malloc( (nodes+1)*sizeof(*senddispl) );
    recvdispl = malloc( (nodes+1)*sizeof(*recvdispl) );
    if ( (!sendarray) || (!recvarray) || (!sendcount) || (!recvcount) || (!senddispl) || (!recvdispl) )
     {
      fprintf( stderr, "Microshmem: no memory in init\n" );
      exit( -1 );
     } 
    for ( i = 0; i <= nodes; i++ )
     {
      senddispl[i] = recvdispl[i] = PORTIONSIZE*i;
      if ( i < nodes ) sendcount[i] = recvcount[i] = 0;
     } 
    return rc;
}
    void shmem_coarray_all( void *section, long sectionsize, void *capointer[] )
{
    int i;
    if ( !section )
     {
      section = malloc( sectionsize );
      if ( !section )
       {
        fprintf( stderr, "Microshmem: no memory in shmem_coarray_all\n" );
        exit( -1 );
       }
     }
    MPI_Allgather(&section, 1, MPI_LONG, capointer, 1, MPI_LONG, shmemcomm);
}
    double shmem_time( void ) { return MPI_Wtime(); }


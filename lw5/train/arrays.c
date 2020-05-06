#include "mpi.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    int N = 100000000;
    double* array = (double*) malloc(sizeof(double) * N);
    srand(time(NULL));
    int i;
    for(i = 0; i < N; ++i)
    {
        array[i] = (rand() % 1000) / (double)1000; 
    }

    double local_sum, global_sum = 0;
    int total_proc_count;
    int curr_proc;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &total_proc_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_proc);

    local_sum = .0;
    for(i = curr_proc; i < N; i += total_proc_count)
    {
        local_sum += array[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(curr_proc == 0)
    {
        printf("%f\n", global_sum);
    }

    MPI_Finalize();
    free(array);

}
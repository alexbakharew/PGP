#include "mpi.h"
#include <math.h>
#include <stdio.h>
double f(double a) {
return (4.0 / (1.0 + a*a));
}
int main(int argc, char *argv) 
{
    int ProcRank, ProcNum, done = 0, n = 0, i;
    double PI25DT = 3.141592653589793238462643;
    double mypi, pi, h, sum, x, t1, t2;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD,&ProcRank);
    while (!done ) 
    { // основной цикл вычислений
        if ( ProcRank == 0) 
        {
            printf("Enter the number of intervals: ");
            scanf("%d",&n);
            t1 = MPI_Wtime();
        }
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (n > 0) 
        { // вычисление локальных сумм
            h = 1.0 / (double) n;
            sum = 0.0;
            for (i = ProcRank + 1; i <= n; i += ProcNum) 
            {
                x = h * ((double)i - 0.5);
                sum += f(x);
            }
            mypi = h * sum;
            // сложение локальных сумм (редукция)
            MPI_Reduce(&mypi,&pi,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            if ( ProcRank == 0 ) 
            { // вывод результатов
                t2 = MPI_Wtime();
                printf("pi is approximately %.16f, Error is %.16f\n",pi, fabs(pi - PI25DT));
                printf("wall clock time = %f\n",t2-t1);
            }
        } 
        else done = 1;
    }
    MPI_Finalize();
}
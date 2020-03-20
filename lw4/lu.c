#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void swap(double* lhs, double* rhs)
{
    int tmp = *lhs;
    *lhs = *rhs;
    *rhs = tmp;
}
void transpose(double** matrix, int size)
{
    for(int i = 0; i < size; ++i)
    {
        for(int j = 0; j < size; ++j)
        {

            if(i < j)
                continue;

            swap(&matrix[i][j], &matrix[j][i]);
        }
    }
}
void print_matrix(double** matrix, int size)
{
    for(int i = 0; i < size; ++i)
    {
        for(int j = 0; j < size; ++j)
        {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}
int main()
{
    int n;
    scanf("%d", &n);
    double** matrix = (double**) malloc(sizeof(double*) * n);
    for(int i = 0; i < n; ++i)
    {
        matrix[i] = (double*) malloc(sizeof(double) * n);
        for(int j = 0; j < n; ++j)
        {
            scanf("%lf", &matrix[i][j]);
        }
    }

    //transpose(matrix, n);
    printf("matrix------------\n");

    print_matrix(matrix, n);

    double** L = (double**) malloc(sizeof(double*) * n);
    double** U = (double**) malloc(sizeof(double*) * n);
    for(int i = 0; i < n; ++i)
    {
        L[i] = calloc(n, sizeof(double));
        U[i] = (double*) malloc(sizeof(double) * n);
        memcpy(U[i], matrix[i], n * sizeof(double));
    }

    for(int col = 0; col < n; ++col)
    {
        int max_ind = col;
        for(int t = col; t < n; ++t)
        {
            if(fabs(U[t][col]) > fabs(U[max_ind][col]))
                max_ind = t;
        }

        if(max_ind != col)
        {
            for(int s = col; s < n; ++s)
            {
                swap(&U[col][s], &U[max_ind][s]);
                swap(&L[col][s], &L[max_ind][s]);
            }
        }

        for(int row = col; row < n; ++row)
        {
            if(row == col)
            {
                L[row][row] = 1.0;
                continue;
            }

            L[row][col] = U[row][col] / U[col][col];

            for(int i = col; i < n; ++i)
            {
                U[row][i] = U[row][i] - L[row][col] * U[col][i];  
            }
        }

    }
    printf("------------\n");
    print_matrix(L, n);
    printf("LU------------\n");
    print_matrix(U, n);
    printf("multiplication------------\n");
    double** res = (double**) malloc(sizeof(double*) * n);
    for(int i = 0; i < n; ++i)
    {
        res[i] = calloc(n, sizeof(double));        
    }

    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j)
        {        
            for(int t = 0; t < n; ++t)
            {
                res[j][i] += L[j][t] * U[t][i];
            }
        }
    }
    print_matrix(res, n);
    return 0;
}
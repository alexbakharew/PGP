#include <stdio.h>
#include <stdlib.h>

void swap(int* lhs, int* rhs)
{
    int tmp = *lhs;
    *lhs = *rhs;
    *rhs = tmp;
}
void transpose(int** matrix, int size)
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
void print_matrix(int** matrix, int size)
{
    for(int i = 0; i < size; ++i)
    {
        for(int j = 0; j < size; ++j)
        {
            printf("%d ", matrix[i][j]);
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
            scanf("%d", &matrix[i][j]);
        }
    }

    //transpose(matrix, n);
    print_matrix(matrix, n);

    double** L = (double**) malloc(sizeof(double*) * n);
    double** U = (double**) malloc(sizeof(double*) * n);
    for(int i = 0; i < n; ++i)
    {
        L[i] = calloc(n, sizeof(double));
        U[i] = calloc(n, sizeof(double));
    }

    for(int col = 0; col < n; ++col)
    {
        int max_ind = col;
        for(int t = col; t < n; ++t)
        {
            if(abs(matrix[t][col]) > abs(matrix[max_ind][col]))
                max_ind = t;
        }
        for(int row = col; row < n; ++row)
        {
            if(row == max_ind)
            {
                L[row][row] = 1.0;
                continue;
            }
            L[row][col] = matrix[row][col] / matrix[max_ind][col];

            for(int i = col; i < n; ++i)
            {
                U[row][col] = matrix[row][i] - L[row][col] * matrix[max_ind][i];  
            }
        }
        

    }



    return 0;
}
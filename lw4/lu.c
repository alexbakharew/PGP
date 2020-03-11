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
    int** matrix = (int**) malloc(sizeof(int*) * n);
    for(int i = 0; i < n; ++i)
    {
        matrix[i] = (int*) malloc(sizeof(int) * n);
        for(int j = 0; j < n; ++j)
        {
            scanf("%d", &matrix[i][j]);
        }
    }

    //transpose(matrix, n);
    print_matrix(matrix, n);

    int** L = (int**) malloc(sizeof(int*) * n);
    int** U = (int**) malloc(sizeof(int*) * n);

    for(int i = 0; i < n; ++i)
    {
        int max_ind = i;
        for(int t = i; t < n; ++t)
        {
            if(abs(matrix[t][i]) > abs(matrix[max_ind][i]))
                max_ind = t;
        }
        
        

    }



    return 0;
}
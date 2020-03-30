#include <stdio.h>
#include <string.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

__host__ void cpu_swap(double* lhs, double* rhs)
{
	double tmp = *lhs;
	*lhs = *rhs;
    *rhs = tmp;    
}

__host__ void cpu_transpose(double* matrix, int size)
{
    for(int i = 0; i < size; ++i)
    {
        for(int j = i + 1; j < size; ++j)
        {
			cpu_swap(&matrix[i * size + j], &matrix[j * size + i]);
        }
    }
}

__global__ void gpu_transpose(double* matrix, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = gridDim.x * blockDim.x;
	int j = 0;
	double temp;
	for (j = idx + 1; j < size; ++j)
	{
        temp = matrix[idx * size + j];
		matrix[idx * size + j] = matrix[j * size + idx];
		matrix[j * size + idx] = temp;
    }
    idx += offsetx;
}
__global__ void gpu_swap(double* matrix, int size, int row_from, int row_to)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = gridDim.x * blockDim.x;
    double tmp; 
    for(int i = idx; i < size; i += offsetx)
    {
        tmp = matrix[(i * size) + row_from];
        matrix[(i * size) + row_from] = matrix[(i * size) + row_to];
        matrix[(i * size) + row_to] = tmp;
    }
}
__host__ void gpu_print_matrix(double* matrix, int size)
{
    for(int i = 0; i < size; ++i)
    {
        for(int j = 0; j < size; ++j)
        {
            printf("%f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

#define THREADS_PER_BLOCK 20
#define BLOCKS_PER_GRID 20

int main()
{
    int n;
    scanf("%d", &n);
    double* matrix = (double*) malloc(sizeof(double) * n * n);
    for(int i = 0; i < n * n; ++i)
    {
		scanf("%lf", &matrix[i]);   
    }

    printf("matrix------------\n");
	gpu_print_matrix(matrix, n);

	double* matrix_dev;
	cudaMalloc(&matrix_dev, sizeof(double) * n * n);
	cudaMemcpy(matrix_dev, matrix, sizeof(double) * n * n, cudaMemcpyHostToDevice);

	gpu_transpose << <32, 32>> > (matrix_dev, n);

    double* L = (double*) malloc(sizeof(double) * n * n);
    thrust::device_ptr<double> p_matrix = thrust::device_pointer_cast(dev_matrix);
    thrust::device_ptr<double> max_pos;   
    int pos;

    int sign = 1;
    for(int row = 0; row < n; ++row)
    {
        max_pos = thrust::max_element(p_matrix + (row * n), p_matrix + (row + 1) * n);
        pos = (int)(max_pos - p_matrix);
        if(pos != row)
        {
            gpu_swap<<<32, 32>>>(matrix_dev, n, row, pos);
        }
        cudaMemcpy(matrix, dev_matrix, sizeof(double) * n * n, cudaMemcpyDeviceToHost);//possible bottle neck
        for(int col = row; col < n; ++col)
        {
            if(col == row)
            {
                L[col][col] = 1.0;
                continue
            }

            if(matrix[row][row] != 0)
                L[col][row] = matrix[col][row] / matrix[row][row];
                
            else
            {
                L[col][row] = 0;
                continue;
            }

            LU<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_matrix, size, col, L[col][row]);

        }
    }
   /* int sign = 1;
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
            sign *= -1;
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
            
            if(U[col][col] != 0)
                L[row][col] = U[row][col] / U[col][col];
            else
            {
                L[row][col] = 0;
                continue;
            }
            

            for(int i = col; i < n; ++i)
            {
                U[row][i] -= L[row][col] * U[col][i];  
            }
        }

    }*/
    //printf("------------\n");
    //print_matrix(L, n);
    //printf("LU------------\n");
    //print_matrix(U, n);
    //printf("multiplication------------\n");
    //
    //double** res = multiplication(L, U, n);
    //print_matrix(res, n);
    //long double d = determinator(L, U, n);
    //printf("%Lf\n", d * sign);
    return 0;
}
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

	double* dev_matrix;
	cudaMalloc(&dev_matrix, sizeof(double) * n * n);
	cudaMemcpy(dev_matrix, matrix, sizeof(double) * n * n, cudaMemcpyHostToDevice);

	gpu_transpose << <32, 32>> > (dev_matrix, n);

	cudaMemcpy(matrix, dev_matrix, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

    thrust::device_ptr<double> p_arr = thrust::device_pointer_cast(dev_matrix);
    thrust::device_ptr<double> res;
    for(int i = 0; i < n - 1; ++i)
    {
        res = thrust::max_element(p_arr + (i * n), p_arr + (i + 1) * n);
        printf("gpu: %d\n", (int)(res - p_arr));
    }


    //double** L = (double**) malloc(sizeof(double*) * n);
    //double** U = (double**) malloc(sizeof(double*) * n);
    //for(int i = 0; i < n; ++i)
    //{
    //    L[i] = (double*) calloc(n, sizeof(double));
    //    U[i] = (double*) malloc(sizeof(double) * n);
    //    memcpy(U[i], matrix[i], n * sizeof(double));
    //}




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
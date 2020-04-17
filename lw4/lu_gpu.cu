#include <stdio.h>
#include <string.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#define THREADS_PER_BLOCK 20
#define BLOCKS_PER_GRID 20

#define CSC(call)  					\
do {								\
	cudaError_t res = call;			\
	if (res != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);					\
	}								\
} while(0)

__host__ void cpu_swap(double* lhs, double* rhs)
{
	double tmp = *lhs;
	*lhs = *rhs;
	*rhs = tmp;
}

__host__ void cpu_transpose(double* matrix, int size)
{
	for (int i = 0; i < size; ++i)
	{
		for (int j = i + 1; j < size; ++j)
		{
			cpu_swap(&matrix[i * size + j], &matrix[j * size + i]);
		}
	}
}   

__host__ void gpu_print_matrix(double* matrix, int size)
{
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			printf("%.1f ", matrix[i * size + j]);
		}
		printf("\n");
	}
}

__global__ void gpu_transpose(double* matrix, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = gridDim.x * blockDim.x;
    double temp;
    int curr_row;
    int curr_col;
    while(idx < size * size)
    {
        curr_row = idx / size;
        curr_col = idx % size;
        if(curr_col > curr_row)
        {
            temp = matrix[curr_row * size + curr_col];
            matrix[curr_row * size + curr_col] = matrix[curr_col * size + curr_row];
            matrix[curr_col * size + curr_row] = temp;
        }
	    idx += offsetx;
        
	}
}

__global__ void gpu_swap(double* matrix, int size, int row_from, int row_to)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = gridDim.x * blockDim.x;
	double tmp;
	for (int i = idx; i < size; i += offsetx)
	{
		tmp = matrix[(i * size) + row_from];
		matrix[(i * size) + row_from] = matrix[(i * size) + row_to];
		matrix[(i * size) + row_to] = tmp;
	}
}

double* multiplication(double* lhs, double* rhs, int n)
{
    double* res = (double*) malloc(sizeof(double) * n * n);

    res = (double*) calloc(n * n, sizeof(double));        


    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j)
        {        
            for(int t = 0; t < n; ++t)
            {
                res[j * n + i] += lhs[j * n + t] * rhs[t * n + i];
            }
        }
    }
    return res;
}

__global__ void gpu_compute_L(double* matrix, double* L, int size, int curr_row)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;//shift from main diagonal
	int offsetx = gridDim.x * blockDim.x;

	for(; idx < size; idx += offsetx)
	{
		if(idx < curr_row)
			continue;

		if(idx == curr_row )
		{
			L[curr_row * size + curr_row] = 1.0;
		}

		else if(matrix[curr_row * size + curr_row] != 0)
		{
			L[curr_row * size + idx] = matrix[curr_row * size + idx] / matrix[curr_row * size + curr_row];
		}
	}
}

__global__ void modify_matrix(double* matrix, double* L, int size, int max_col)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = gridDim.x * blockDim.x;
	int curr_row;
	int curr_col;
	for(; idx < size * size; idx += offsetx)
	{
		curr_row = idx / size;
		curr_col = idx % size;
		if(curr_col == max_col)
			continue;
		else
		{
			//matrix[idx] -= L[max_col * size + curr_col] * matrix[curr_row * size + max_col];
			matrix[curr_row * size + curr_col] -= L[max_col * size + curr_col] * matrix[curr_row * size + max_col];
			
		}
	}
}
int main()
{
	int n;
	scanf("%d", &n);
	double* matrix = (double*)malloc(sizeof(double) * n * n);
	for (int i = 0; i < n * n; ++i)
	{
		scanf("%lf", &matrix[i]);
	}
	double* matrix_dev;
	CSC(cudaMalloc(&matrix_dev, sizeof(double) * n * n));
	CSC(cudaMemcpy(matrix_dev, matrix, sizeof(double) * n * n, cudaMemcpyHostToDevice));

	// printf("matrix------------\n");
	// gpu_print_matrix(matrix, n);
	// printf("\n");

	gpu_transpose << <32, 32 >> > (matrix_dev, n);

	cudaMemcpy(matrix, matrix_dev, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
	printf("transposed matrix------------\n");
	gpu_print_matrix(matrix, n);

	double* L = (double*) calloc(n * n, sizeof(double));
	double* L_dev;
	CSC(cudaMalloc(&L_dev, sizeof(double) * n * n));
	CSC(cudaMemcpy(L_dev, L, sizeof(double) * n * n, cudaMemcpyHostToDevice));

	int* P = (int*) malloc(sizeof(int) * n);
	
	// printf("jopa");

	// thrust::device_ptr<double> max_elem;
	int pos_of_max;
	int sign = 1;
	// getchar();

	for (int row = 0; row < n; ++row)
	{
		thrust::device_ptr<double> p_matrix = thrust::device_pointer_cast(matrix_dev);
		thrust::device_ptr<double> max_elem = thrust::max_element(p_matrix + (row * n) + row, p_matrix + ((row + 1) * n));
		pos_of_max = (int)(max_elem - p_matrix) % n;
		//printf("%d\n", pos_of_max);
		//P[row] = pos_of_max;

		if(row != pos_of_max)
		{
			sign *= -1;
			gpu_swap<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>> (matrix_dev, n, row, pos_of_max);
			CSC(cudaMemcpy(matrix, matrix_dev, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
			printf("U after swap-----\n");
			gpu_print_matrix(matrix, n);
			
		}

		gpu_compute_L << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (matrix_dev, L_dev, n, row);
		CSC(cudaMemcpy(L, L_dev, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
		printf("L-----\n");
		gpu_print_matrix(L, n);
		getchar();

		modify_matrix <<< BLOCKS_PER_GRID, THREADS_PER_BLOCK >>> (matrix_dev, L_dev, n, row);
		CSC(cudaMemcpy(matrix, matrix_dev, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
		printf("U-----\n");
		gpu_print_matrix(matrix, n);
		printf("\n\n");
		getchar();
		printf("=====================================\n");


	}
	CSC(cudaMemcpy(L, L_dev, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
	CSC(cudaMemcpy(matrix, matrix_dev, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
	printf("L-----\n");
	gpu_print_matrix(L, n);
	printf("matrix-----\n");
	gpu_print_matrix(matrix, n);
	
	double det = 1;

	for(int i = 0; i < n; ++i)
	{
		//printf("p[i] = %d ", P[i]);
		printf("L[i] = %lg U[i] = %lg\n", L[i * n + i], matrix[i * n + i]);
		det *= matrix[i * n + i] * L[i * n + i];
	}
	printf("\n");
	printf("det = %f\n", det * sign);
	//printf("matrix-----\n");
	//gpu_print_matrix(matrix, n);


}
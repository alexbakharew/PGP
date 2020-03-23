#include <stdio.h>
#include <string.h>

#define CSC(call)  					\
do {								\
	cudaError_t res = call;			\
	if (res != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);					\
	}								\
} while(0)

#define THREADS_PER_BLOCK 20
#define BLOCKS_PER_GRID 20

__global__ void scalar(const int* arr1, const int* arr2, const int size, int* res)
{
    __shared__ int cache[THREADS_PER_BLOCK]; 
    int offsetx = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int temp = 0;
    while(tid < size)
    {
        temp += arr1[tid] * arr2[tid];
        tid += offsetx;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    
}
int main()
{
    int size;
    scanf("%d", &size);
    int* arr1 = (int*) malloc(size * sizeof(int));
    int* arr2 = (int*) malloc(size * sizeof(int));

    for(int i = 0; i < size; ++i)
    {
        scanf("%d", &arr1[i]);
    }
    
    for(int i = 0; i < size; ++i)
    {
        scanf("%d", &arr2[i]);
    }

    int* dev_arr1;
    int* dev_arr2;
    int* dev_res;
    
    CSC(cudaMalloc(&dev_arr1, sizeof(int) * size));
    CSC(cudaMalloc(&dev_arr2, sizeof(int) * size));
    CSC(cudaMalloc(&dev_res, sizeof(int) * size));
    
    CSC(cudaMemcpy(dev_arr1, arr1, sizeof(int) * size, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(dev_arr2, arr2, sizeof(int) * size, cudaMemcpyHostToDevice));

    scalar<<<20, 20>>>(dev_arr1, dev_arr2, size, dev_res);

    int* res = (int*) malloc(size * sizeof(int));
    cudaMemcpy(res, dev_res, sizeof(int) * size, cudaMemcpyDeviceToHost);

    long long int scalar_mult = 0;

    for(int i = 0; i < size; ++i)
    {
        scalar_mult += res[i];
    }
    printf("%llu\n", scalar_mult);
    return 0;

}
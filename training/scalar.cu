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

__global__ void scalar(const int* arr1, const int* arr2, int size, int* res)
{
    int offsetx = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("jopa\n");
    while(tid < size)
    {
        res[tid] = arr1[tid] * arr2[tid];
        tid += offsetx;
        printf("%d ", res[tid]);
    }
}
int main()
{
    int size;
    scanf("%d", &size);
    int* arr1 = (int*) malloc(size * sizeof(int));
    int* arr2 = (int*) malloc(size * sizeof(int));
    int* res = (int*) malloc(size * sizeof(int));
    for(int i = 0; i < size; ++i)
    {
        scanf("%d", &arr1[i]);
    }
    
    for(int i = 0; i < size; ++i)
    {
        scanf("%d", &arr2[i]);
    }

    // for(int i = 0; i < size; ++i)
    // {
    //     printf("%d %d\n", arr1[i], arr2[i]);
    // }
    int* dev_arr1;
    int* dev_arr2;
    int* dev_res;
    
    CSC(cudaMalloc(&dev_arr1, sizeof(int) * size));
    CSC(cudaMalloc(&dev_arr2, sizeof(int) * size));
    CSC(cudaMalloc(&dev_res, sizeof(int) * size));
    
    cudaMemcpy(dev_arr1, arr1, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_arr2, arr2, sizeof(int) * size, cudaMemcpyHostToDevice);
    
    scalar<<<1024, 1024>>>(dev_arr1, dev_arr2, size, dev_res);

    cudaMemcpy(res, dev_res, sizeof(int) * size, cudaMemcpyDeviceToHost);

    long long int scalar_mult = 1;

    for(int i = 0; i < size; ++i)
    {
        scalar_mult *= res[i];
        //printf("%d ", res[i]);
    }
    printf("%llu\n", scalar_mult);
    return 0;
}
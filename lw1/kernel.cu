#include <iostream>
#include <iomanip>
#include <time.h>

using ll = long long;
#define CSC(call)  					\
do {								\
	cudaError_t res = call;			\
	if (res != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);					\
	}								\
} while(0)



void RunOnCPU(const double* arr1, const double* arr2, const ll size)
{
	int* result = (int*) malloc(size * sizeof(int));
	for (ll i = 0; i < size; ++i)
	{
		result[i] = (arr1[i] > arr2[i] ? arr1[i] : arr2[i]);
	}
}

__global__ void kernel(const double* arr1, const double* arr2, double* result, const ll size)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	while (idx < size)
	{
		result[idx] = (arr1[idx] > arr2[idx] ? arr1[idx] : arr2[idx]);
		idx += offset;
	}
}

int main() 
{
	ll n;
	double* arr1;
	double* arr2;
	scanf("%lli", &n);
	if (n < 0)
	{
		printf("ERROR: negative size of vectors. Exit\n");
		return 0;
	}
	arr1 = (double*)(malloc(n * sizeof(double)));
	arr2 = (double*)(malloc(n * sizeof(double)));
	for (int i = 0; i < n; ++i)
	{
		scanf("%lf", &arr1[i]);
	}

	for (int i = 0; i < n; ++i)
	{
		scanf("%lf", &arr2[i]);
	}
	clock_t tStart = clock();
	RunOnCPU(arr1, arr2, n);
	printf("CPU time : %.2fms\n", (double)(clock() - tStart) * 1000 /CLOCKS_PER_SEC);
	double* dev_arr1;
	double* dev_arr2;
	double* result_on_gpu;

	CSC(cudaMalloc(&dev_arr1, sizeof(double) * n));
	CSC(cudaMalloc(&dev_arr2, sizeof(double) * n));
	CSC(cudaMalloc(&result_on_gpu, sizeof(double) * n));

	CSC(cudaMemcpy(dev_arr1, arr1, sizeof(double) * n, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(dev_arr2, arr2, sizeof(double) * n, cudaMemcpyHostToDevice));

	//calc time
	tStart = clock();
	kernel << <1024, 1024 >> > (dev_arr1, dev_arr2, result_on_gpu, n);
	printf("GPU time : %.5fms\n", (double)(clock() - tStart) * 1000 /CLOCKS_PER_SEC);
	double* result_on_cpu = (double*)malloc(sizeof(double) * n);

	CSC(cudaMemcpy(result_on_cpu, result_on_gpu, sizeof(double) * n, cudaMemcpyDeviceToHost));
	CSC(cudaGetLastError());

	// for (ll i = 0; i < n; ++i)
	// {
	// 	std::cout << std::fixed << std::scientific;

	// 	std::cout << std::setprecision(10) << result_on_cpu[i];
	// 	if (i < n - 1)
	// 		std::cout << " ";
	// 	else
	// 		std::cout << std::endl;
	// }

	free(arr1);
	free(arr2);
	free(result_on_cpu);
	CSC(cudaFree(dev_arr1));
	CSC(cudaFree(dev_arr2));
	CSC(cudaFree(result_on_gpu));

	return 0;
}

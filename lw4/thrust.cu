#include <stdio.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

struct type {
	int key;
	int value;
}; 

struct comparator {
	__host__ __device__ bool operator()(type a, type b) {
		return a.key < b.key; 
	}
};

int main() {
	srand(time(NULL));
	comparator comp;
	int i, i_max = -1, n = 100000;
	type *arr = (type *)malloc(sizeof(type) * n);
	for(i = 0; i < n; i++) {
		arr[i].key = rand();
		arr[i].value = rand();
		if (i_max == -1 || comp(arr[i_max], arr[i]))
			i_max = i;
	}
	type *dev_arr;
	cudaMalloc(&dev_arr, sizeof(type) * n);
	cudaMemcpy(dev_arr, arr, sizeof(type) * n, cudaMemcpyHostToDevice); 
	
	thrust::device_ptr<type> p_arr = thrust::device_pointer_cast(dev_arr);
	thrust::device_ptr<type> res = thrust::max_element(p_arr, p_arr + n,comp);

	printf("cpu: %d\ngpu: %d\n", i_max, (int)(res - p_arr));
	printf("%d %d", (int)res, (int)p_arr);
	cudaFree(dev_arr);
	free(arr);
	return 0;
}

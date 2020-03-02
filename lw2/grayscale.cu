#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NAME_LEN 128

#define CSC(call)  		\
do {								\
	cudaError_t res = call;			\
	if (res != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);					\
	}								\
} while(0)

#define MIN(a,b) a < b ? a : b
#define MAX(a,b) a > b ? a : b

texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(uchar4 *dst, int w, int h) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	uchar4 p;
	for(x = idx; x < w; x += offsetx) 
	{
		for(y = idy; y < h; y += offsety) 
		{
			p = tex2D(tex, x, y);
			unsigned char res = (0.299 * p.x) + (0.587 * p.y) + (0.114 * p.z);

			dst[y * w + x] = make_uchar4(res, res, res, p.w);
		}
	}
}

int main(int argc, char* argv[])
{
	char input[NAME_LEN];
	char output[NAME_LEN];

	scanf("%s", input);
	scanf("%s", output);

	int width, height;
	FILE *in = fopen(input, "rb");
	if(in == NULL)
	{
		printf("Error while opening input file\n");
		exit(-1);
	}
	fread(&width, sizeof(int), 1 , in);
	fread(&height, sizeof(int), 1 , in);
	uchar4 *image = (uchar4*)malloc(sizeof(uchar4) * width * height);
	fread(image, sizeof(uchar4), width * height, in);
	fclose(in);

	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, width, height));
	CSC(cudaMemcpyToArray(arr, 0, 0, image, sizeof(uchar4) * height * width, cudaMemcpyHostToDevice));

	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false; 

	uchar4* new_image = (uchar4*)malloc(sizeof(uchar4) * width * height);

	CSC(cudaBindTextureToArray(tex, arr, ch));
	uchar4 *dev_data;
	CSC(cudaMalloc(&dev_data, sizeof(uchar4) * height * width));

	cudaEvent_t start, end;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));
	kernel<<<dim3(32, 32), dim3(32, 32)>>>(dev_data, width, height);
	CSC(cudaGetLastError());

	CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
	float t;
	CSC(cudaEventElapsedTime(&t, start, end));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(end));
	printf("GPU time = %.2fms\n", t);

	CSC(cudaMemcpy(new_image, dev_data, sizeof(uchar4) * height * width, cudaMemcpyDeviceToHost));

	clock_t start_time = clock();
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
			int pos = i * width + j;
			long res = MAX(MAX(image[pos].x, image[pos].y), image[pos].z) + MIN(MIN(image[pos].x, image[pos].y), image[pos].z);
			res /= 2;
			new_image[pos].x = res;
			new_image[pos].y = res;
			new_image[pos].z = res;
			new_image[pos].w = image[pos].w;
        }
	}
	printf("CPU time = %.2fms\n", (double)(clock() - start_time) * 1000 /CLOCKS_PER_SEC);

	FILE *out = fopen(output, "wb");
	if(out == NULL)
	{
		printf("Error while opening output file\n");
		free(image);
		free(new_image);
		exit(-1);
	}
	fwrite(&width, sizeof(int), 1, out);
	fwrite(&height, sizeof(int), 1, out);
	fwrite(new_image, sizeof(uchar4), width * height, out);

	cudaUnbindTexture(tex);
	cudaFreeArray(arr);
	cudaFree(dev_data);
	fclose(out);
	free(image);
	free(new_image);
}
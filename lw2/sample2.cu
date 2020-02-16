#include <stdio.h>


texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(uchar4 *dst, int w, int h) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	uchar4 p;
	for(x = idx; x < w; x += offsetx) 
		for(y = idy; y < h; y += offsety) {
			p = tex2D(tex, x, y);
			dst[y * w + x] = make_uchar4(~p.x, ~p.y, ~p.z, p.w);
		}
}

int main() {
	int w, h;
	FILE *in = fopen("in.data", "rb");
	fread(&w, sizeof(int), 1 , in);
	fread(&h, sizeof(int), 1 , in);
	uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * h * w);
	fread(data, sizeof(uchar4), h * w, in);
	fclose(in);

	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	cudaMallocArray(&arr, &ch, w, h);
	cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice);

	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false; 

	cudaBindTextureToArray(tex, arr, ch);
	uchar4 *dev_data;
	cudaMalloc(&dev_data, sizeof(uchar4) * h * w);
	kernel<<<dim3(16, 16), dim3(16, 16)>>>(dev_data, w, h);
	cudaMemcpy(data, dev_data, sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost);

	FILE *out = fopen("out.data", "wb");
	fwrite(&w, sizeof(int), 1, out);
	fwrite(&h, sizeof(int), 1, out);
	fwrite(data, sizeof(uchar4), w * h, out);
	fclose(out);

	cudaUnbindTexture(tex);
	cudaFreeArray(arr);
	cudaFree(dev_data);
	free(data);

	return 0;
}

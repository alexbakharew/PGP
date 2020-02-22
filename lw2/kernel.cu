#include <cuda.h>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define HANDLE_ERROR(call) \
{ \
	cudaError_t err = call; \
	\
	if (err != cudaSuccess) \
	{ \
		fprintf(stderr, "ERROR: CUDA failed in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		exit(0); \
	} \
}

typedef uchar4 Pixel;

class Image
{
public:
	int width;
	int height;
	Pixel *pixels;

	Image()
	{

	}

	Image(int w, int h)
	{
		width = w;
		height = h;
	}

	Image(std::string filename)
	{
		FILE* file;
		if ((file = fopen(filename.c_str(), "rb")) == NULL) {
			std::cout << "Can't load image from file" << std::endl;
			exit(1);
		}

		fread(&width, 1, sizeof(int), file);
		fread(&height, 1, sizeof(int), file);
		//fread(&height, sizeof(int), 1, file);
		printf("%i %i\n", width, height);
		pixels = new Pixel[width*height];
		fread(pixels, sizeof(Pixel), width * height, file);

		fclose(file);
	}

	void WriteToFile(std::string filename);
	friend std::ostream& operator<<(std::ostream& os, const Image image);
	int GetSize();
	std::string toString();

	void addClasters(int *claster_by_pixel_index);
};

void Image::WriteToFile(std::string filename)
{
	FILE* file = fopen(filename.c_str(), "wb");

	fwrite(&width, sizeof(width), 1, file);
	fwrite(&height, sizeof(height), 1, file);
	fwrite(pixels, sizeof(Pixel), width * height, file);
	fclose(file);
}

std::ostream& operator<<(std::ostream& os, const Image image)
{
	os << "width = " << image.width << std::endl;
	os << "height = " << image.height << std::endl;
	os << "pixels: " << std::endl;
	for (int i = 0; i < image.height*image.width; i++) {
		os << std::hex << (int)image.pixels[i].x << " " << (int)image.pixels[i].y << " " << (int)image.pixels[i].z << " " << (int)image.pixels[i].w << std::endl;
	}

	return os;
}

std::string Image::toString() {
	std::stringstream stream;
	stream << width << " " << height << "\n";
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int k = i*width + j;
			stream << std::hex << std::setfill('0') << std::setw(2) << (int)pixels[k].x << std::setfill('0') << std::setw(2) << (int)pixels[k].y << std::setfill('0') << std::setw(2) << (int)pixels[k].z << std::setfill('0') << std::setw(2) << (int)pixels[k].w << " ";
		}
		stream << "\n";
	}

	return stream.str();
}

int Image::GetSize()
{
	return width*height * sizeof(Pixel) + sizeof(int) + sizeof(int);
}

void Image::addClasters(int *claster_by_pixel_index) {
	for (int i = 0; i < height*width; i++) {
		pixels[i].w = claster_by_pixel_index[i];
	}
}

typedef const char* CString;
typedef unsigned char Byte;
typedef uchar4 Pixel;
typedef texture<Pixel, 2, cudaReadModeElementType> Texture2D;

cudaArray* g_arr;
Texture2D g_tex;

void imageCreateTexture(Image* image)
{
	int w = image->width;
	int h = image->height;

	g_tex.channelDesc = cudaCreateChannelDesc<Pixel>();
	g_tex.addressMode[0] = cudaAddressModeClamp;
	g_tex.addressMode[1] = cudaAddressModeClamp;
	g_tex.filterMode = cudaFilterModePoint;
	g_tex.normalized = false;

	HANDLE_ERROR(cudaMallocArray(&g_arr, &g_tex.channelDesc, w, h));
	HANDLE_ERROR(cudaMemcpyToArray(g_arr, 0, 0, image->pixels, sizeof(Pixel) * w * h, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaBindTextureToArray(g_tex, g_arr, g_tex.channelDesc));
}

__device__ double filterGrayScale(Pixel* pixel);
__global__ void filterSobelKernel(Pixel* pixels, int w, int h);

__constant__ int g_filter[6];

__device__ double filterGrayScale(Pixel* pixel)
{
	return pixel->x * 0.299 + pixel->y * 0.587 + pixel->z * 0.114;
}

__global__ void filterPrevitt(Pixel* pixels, int w, int h)
{
	int tY = blockIdx.y * blockDim.y + threadIdx.y;
	int tX = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetY = gridDim.y * blockDim.y;
	int offsetX = gridDim.x * blockDim.x;

	for (int i = tY; i < h; i += offsetY)
	{
		for (int j = tX; j < w; j += offsetX)
		{
			double gx = 0.0;
			double gy = 0.0;
			Pixel pixel;

			for (int k = 0; k < 3; ++k)
			{
				int row = i + k - 1;
				int col0 = j - 1;
				int col2 = j + 1;
				int col = j + k - 1;
				int row0 = i - 1;
				int row2 = i + 1;

				pixel = tex2D(g_tex, col0, row);
				gx += g_filter[k] * filterGrayScale(&pixel);
				pixel = tex2D(g_tex, col2, row);
				gx += g_filter[k + 3] * filterGrayScale(&pixel);
				pixel = tex2D(g_tex, col, row0);
				gy += g_filter[k] * filterGrayScale(&pixel);
				pixel = tex2D(g_tex, col, row2);
				gy += g_filter[k + 3] * filterGrayScale(&pixel);
			}

			Byte gm = (Byte)min((int)sqrt(gx * gx + gy * gy), (int)0xFF);
			int offset = i * w + j;

			pixels[offset].x = gm;
			pixels[offset].y = gm;
			pixels[offset].z = gm;
			pixels[offset].w = 0;
		}
	}
}

void Start(Image *image) {
	Pixel* dPixels;
	//int filter[] = { -1, -2, -1, 1, 2, 1 };
	int filter[] = { -1, -1, -1, 1, 1, 1 };
	imageCreateTexture(image);

	int size = sizeof(Pixel) * image->width * image->height;

	HANDLE_ERROR(cudaMalloc(&dPixels, size));
	HANDLE_ERROR(cudaMemcpyToSymbol(g_filter, filter, sizeof(filter)));

	dim3 gridSize(32, 32);
	dim3 blockSize(32, 32);

	filterPrevitt << <gridSize, blockSize >> >(dPixels, image->width, image->height);

	HANDLE_ERROR(cudaMemcpy(image->pixels, dPixels, size, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(dPixels));

}

int main(void)
{
	std::string input_filename;
	std::string output_filename;

	std::cin >> input_filename >> output_filename;

	Image image(input_filename);

	Start(&image);

	image.WriteToFile(output_filename);

	return 0;
}
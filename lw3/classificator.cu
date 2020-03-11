#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define NAME_LEN 128
#define CUDA
#ifndef CUDA
typedef struct 
{
    unsigned char x, y, z, w;
}uchar4;
#endif

__constant__ uchar3 dev_avg[3];

__global__ void Classificator(uchar4* dev_image, int width, int height)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y;
    uchar4 p;
    for(x = idx; x < height; x += offsetx)
    {
        for(y = idy; y < width; y += offsety)
        {
            int pos = x * width + y;
            p = dev_image[pos];
            int c1 = (p.x - dev_avg[0].x) * (p.x - dev_avg[0].x) + ((p.y - dev_avg[0].y) * (p.y - dev_avg[0].y)) + ((p.z - dev_avg[0].z) * (p.z - dev_avg[0].z));
            int c2 = (p.x - dev_avg[1].x) * (p.x - dev_avg[1].x) + ((p.y - dev_avg[1].y) * (p.y - dev_avg[1].y)) + ((p.z - dev_avg[1].z) * (p.z - dev_avg[1].z));
            int c3 = (p.x - dev_avg[2].x) * (p.x - dev_avg[2].x) + ((p.y - dev_avg[2].y) * (p.y - dev_avg[2].y)) + ((p.z - dev_avg[2].z) * (p.z - dev_avg[2].z));
            c1 *= -1;
            c2 *= -1;
            c3 *= -1;

            int result = max(c1, max(c2, c3));
            if(result == c1)
                p.w = 0;
            else if(result == c2)
                p.w = 1;
            else
                p.w = 2;
            dev_image[pos] = p;
        }
    }
}
void RunOnCpu(uchar4* image, int width, int height)
{
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            int pos = i * width + j;
            uchar4 p = image[pos];
            // printf("[%d %d] %d %d %d    ",i, j, p.x, p.y, p.z);
            int c1 = (p.x - 255) * (p.x - 255) + (p.y * p.y) + (p.z * p.z);
            int c2 = (p.x * p.x) + (p.y - 255) * (p.y - 255) + (p.z * p.z);
            int c3 = (p.x * p.x) + (p.y * p.y) + (p.z - 255) * (p.z - 255);
            c1 *= -1;
            c2 *= -1;
            c3 *= -1;

            int result = max(c1, max(c2, c3));
            // printf("c1=%d c2=%d c3=%d result = %d    ", c1, c2, c3, result);
            if(result == c1)
                p.w = 0;
            else if(result == c2)
                p.w = 1;
            else
                p.w = 2;
            image[pos] = p;
            // printf("%d\n", p.w);
        }
    }
}
int main()
{
    char input[NAME_LEN];
    char output[NAME_LEN];

    scanf("%s", input);
    scanf("%s", output);

    int width, height;
    FILE* in = fopen(input, "rb");

    fread(&width, sizeof(int), 1, in);
    fread(&height, sizeof(int), 1, in);

    uchar4* image = (uchar4*) malloc(sizeof(uchar4) * width * height);
    fread(image, sizeof(uchar4), width * height, in);
    fclose(in);

    uchar4* dev_image;
    cudaMalloc(&dev_image, sizeof(uchar4) * width * height);
    cudaMemcpy(dev_image, image, sizeof(uchar4) * width * height, cudaMemcpyHostToDevice);

    uchar3* avg = (uchar3*) malloc(sizeof(uchar3) * 3);
    avg[0] = make_uchar3(255, 0, 0);
    avg[1] = make_uchar3(0, 255, 0); 
    avg[2] = make_uchar3(0, 0, 255);
    
    cudaMemcpyToSymbol(dev_avg, avg, sizeof(uchar3) * 3, 0, cudaMemcpyHostToDevice);

    Classificator<<<dim3(32, 32), dim3(32, 32)>>>(dev_image, width, height);
    //RunOnCpu(image, width, height);
    cudaMemcpy(image, dev_image, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);

    FILE* out = fopen(output, "wb");
    fwrite(&width, sizeof(int), 1, out);
    fwrite(&height, sizeof(int), 1, out);
    fwrite(image, sizeof(uchar4), width * height, out);
    fclose(out);
    free(image);
    cudaFree(dev_image);
}
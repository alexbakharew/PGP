#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define NAME_LEN 128

#ifndef CUDA
typedef struct 
{
    unsigned char x, y, z, w;
}uchar4;
#endif

static inline int max(int a, int b)
{
    return a > b ? a : b;
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

    uchar4* new_image = (uchar4*) malloc(sizeof(uchar4) * width * height);
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
            new_image[pos] = p;
            // printf("%d\n", p.w);
        }
    }

    FILE* out = fopen(output, "wb");
    fwrite(&width, sizeof(int), 1, out);
    fwrite(&height, sizeof(int), 1, out);
    fwrite(new_image, sizeof(uchar4), width * height, out);
    fclose(out);
    free(image);
    free(new_image);
}
#include <stdio.h>
#include <stdlib.h>
#define NAME_LEN 32

//#define CUDA
#ifndef CUDA
struct uchar4
{
	int r;
	int g;
	int b;
	int alpha;
};
#endif
#define MAX(A,B,C) A > B ? (A > C ? A : C) : (B > C ? B : C) 
#define MIN(A,B,C) A < B ? (A < C ? A : C) : (B < C ? B : C) 

int main()
{
	char input[NAME_LEN];
	char output[NAME_LEN];
	int width, height;

	scanf("%s", input);
	scanf("%s", output);

	FILE* input_file = fopen(input, "r");
	
	fscanf(input_file, "%d %d\n", &width, &height);

	uchar4* image = (uchar4*) malloc(sizeof(uchar4) * width * height);
	
    for(int i = 0; i < width * height; ++i)
	{
		char buff[3];

		fread(buff, sizeof(char), 2, input_file);
		buff[2] = '\0';
		image[i].r = (int)strtol(buff, NULL, 16);

		fread(buff, sizeof(char), 2, input_file);
		buff[2] = '\0';
		image[i].g = (int)strtol(buff, NULL, 16);

		fread(buff, sizeof(char), 2, input_file);
		buff[2] = '\0';
		image[i].b = (int)strtol(buff, NULL, 16);

		fread(buff, sizeof(char), 2, input_file);
		buff[2] = '\0';
		image[i].alpha = (int)strtol(buff, NULL, 16);

		fgetc(input_file);//skip space or new line
	}
	fclose(input_file);
	uchar4* new_image = (uchar4*)malloc(sizeof(uchar4) * width * height);
	
    for(int i = 0; i < width; ++i)
    {
        for(int j = 0; j < height; ++j)
        {
			int pos = i * width + j;
			int res = MAX(image[pos].r, image[pos].g, image[pos].b) + MIN(image[pos].r, image[pos].g, image[pos].b);
			res /= 2;
			new_image[pos].r = res;
			new_image[pos].g = res;
			new_image[pos].b = res;
			new_image[pos].alpha = image[pos].alpha;
        }
    }
	FILE* output_file = fopen(output, "w");
	fprintf(output_file, "%d %d\n", width, height);
	//fwrite(&width, sizeof(int), 1, output_file);;
	//fwrite(&height, sizeof(int), 1, output_file);
	//fwrite(new_image, sizeof(uchar4) + 1, height * width, output_file);
    char space;
	for(int i = 0; i < width * height; ++i)
	{
		if((i + 1) % width == 0)
			space = '\n';
		else
			space = ' ';
		fprintf(output_file, "%02X%02X%02X%02X%c", new_image[i].r, new_image[i].g, new_image[i].b, new_image[i].alpha, space);
	}
}
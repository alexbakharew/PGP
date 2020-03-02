// Bilinear_Interpolation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define NAME_LEN 32

int main(int argc, char* argv[])
{
	char input[NAME_LEN];
	char output[NAME_LEN];
	int new_width, new_height;
	strcpy(input, argv[1]);
	strcpy(output, argv[2]);
	new_width = atoi(argv[3]);
	new_height = atoi(argv[4]);


	FILE* input_file = fopen(input, "rb");
	int old_width, old_height;
	
	if (fread(&old_width, sizeof(int), 1, input_file) != 1)
	{
		printf("Error while reading data from file on line %d", __LINE__);
	}

	if (fread(&old_height, sizeof(int), 1, input_file) != 1)
	{
		printf("Error while reading data from file on line %d", __LINE__);
	}

	uchar4* image = (uchar4*) malloc(sizeof(uchar4) * old_width * old_height);
	if (fread(image, sizeof(uchar4), old_width * old_height, input_file) != old_width * old_height)
	{
		printf("Error while reading data from file on line %d", __LINE__);
	}

	uchar4* new_image = (uchar4*)malloc(sizeof(uchar4) * new_width * new_height);

	for (int i = 0; i < new_height; ++i)
	{
		for (int j = 0; j < new_width; ++j)
		{
			float a = (i + 0.5) * (old_width / new_width) - 0.5;
			float b = (i + 0.5) * (old_height / new_width) - 0.5;

			int ii = floor(a);
			int jj = floor(b);
			float x = a - ii;
			float y = b - jj;
			
			new_image[i * new_width + j] = (1 - x) * (1 - y) * image[ii * old_width + jj] + 
				(1 - x) * y * image[ii * old_width + jj + 1] + x * (1 - y) * image[(ii + 1) * old_width + jj] +
				x * y * image[(ii + 1) * old_width + jj + 1];
		}
	}

	FILE *out = fopen(output, "wb");
	fwrite(&new_width, sizeof(int), 1, out);
	fwrite(&new_height, sizeof(int), 1, out);
	fwrite(data, sizeof(uchar4), new_height * new_width, out);
	fclose(out);
}
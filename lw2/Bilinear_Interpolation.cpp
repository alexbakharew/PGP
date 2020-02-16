// Bilinear_Interpolation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <stdio.h>
#include <stdlib.h>
#define NAME_LEN 32
struct uchar4
{
	char r;
	char g;
	char b;
	char alpha;
};
int main()
{
	char input[NAME_LEN];
	char output[NAME_LEN];
	int new_width, new_height;

	scanf("%s", input);
	scanf("%s", output);
	scanf("%d %d", &new_width, &new_height);

	FILE* input_file = fopen(input, "r");
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
	
	double initial_x_old = (1 / old_width / 2);
	double initial_y_old = (1 / old_height / 2);

	double initial_x_new = (1 / new_width / 2);
	double initial_y_new = (1 / new_height / 2);

	double step_x_old = 1 / old_width;
	double step_y_old = 1 / old_height;

	double step_x_new = 1 / new_width;
	double step_y_new = 1 / new_height;

	for (int i = 0; i < new_width; ++i)
	{
		for (int j = 0; j < new_height; ++j)
		{
			double curr_x = initial_x_new + i * step_x_new;
			double curr_y = initial_y_new + j * step_y_new;

			int x_ind = curr_x / step_x_old;
			int y_ind = curr_y / step_y_old;

		}
	}
}
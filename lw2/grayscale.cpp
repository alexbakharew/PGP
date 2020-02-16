#include <stdio.h>
#include <stdlib.h>
#define NAME_LEN 32
struct uchar4
{
	int r;
	int g;
	int b;
	int alpha;
};
int main()
{
	char input[NAME_LEN];
	char output[NAME_LEN];
	int width, height;

	scanf("%s", input);
	scanf("%s", output);

	FILE* input_file = fopen(input, "r");
	
	fscanf(input_file, "%d %d\n", &width, &height);

	printf("%d %d\n", width, height);
	unsigned int* image = (unsigned int*) malloc(sizeof(unsigned int) * width * height);
	
    if (fread(image, sizeof(unsigned int) * width * height, 1, input_file) != width * height)
	{
		printf("Error while reading data from file on line %d\n", __LINE__);
	}
    printf("here\n");
	//uchar4* new_image = (uchar4*)malloc(sizeof(uchar4) * width * height);
	
    for(int i = 0; i < width; ++i)
    {
        for(int j = 0; j < height; ++j)
        {
			printf("%d ", image[i * width + j]);
        }
		printf("\n");
    }
    fclose(input_file);
}
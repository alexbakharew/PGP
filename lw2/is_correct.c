#include <stdio.h>

int main() 
{
	int w = 0, h = 0;
	FILE *in = fopen("in.txt", "r");
	fread(&w, sizeof(int), 1 , in);
	fread(&h, sizeof(int), 1 , in);
    printf("%d %d\n", w, h);
	fclose(in);
}
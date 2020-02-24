#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main (int argc, char* argv[]) 
{
	char name[128];
	char buff[32];
	strcpy(name, argv[1]);
	FILE* file = fopen(argv[1], "rb");
	if(file == NULL)
	{
		printf("Jopa\n");
	}
	else
	{
		fread(buff, sizeof(char), 32, file);
		printf("%s\n", buff);
	}
	return(0);
}
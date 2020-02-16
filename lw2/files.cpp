#include <stdio.h>
#include <stdlib.h>

int main()
{
    FILE* file = fopen("in.data", "rb");
    int w,h;
    char buffer[1024];
    fscanf(file, "%d %d\n", &w, &h);
    printf("%d %d\n", w, h);
    fread(buffer, sizeof(char), 1024, file);
    printf("%s\n", buffer);
}
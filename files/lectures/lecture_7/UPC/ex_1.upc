#include <upc_relaxed.h>
#include <stdio.h>
int main()
{
	printf("Thread %d of %d: Hello World\n", MYTHREAD,THREADS);
        printf("done.\n");
	return 0;
}


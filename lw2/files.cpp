#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main () {
   int val;
   char hex[4];
   
   strcpy(hex, "AAA");
    hex[3] = '\0';
   val = (int)strtol(hex, NULL, 16);
   printf("String value = %s, Int value = %d\n", hex, val);

   return(0);
}
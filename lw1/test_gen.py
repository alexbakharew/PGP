import sys
import random

if len(sys.argv) != 2:
    exit(-1)

n = int(sys.argv[1])
with open("01.t", "w") as input:
    input.write(str(n) + "\n")
    for _ in range(n):
        input.write(str(random.randrange(-1000000, 1000000)))
        input.write(" ")
    input.write("\n")
    
    for _ in range(n):
        input.write(str(random.randrange(-1000000, 1000000)))
        input.write(" ")
    input.write("\n")
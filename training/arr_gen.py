import random

with open("in.txt", "w") as input:
    length = random.randrange(10000, 100000);
    input.write("{}\n".format(length))
    for _ in range(length):
        input.write("{} ".format(random.randrange(1, 5)))
    input.write("\n")
    for _ in range(length):
        input.write("{} ".format(random.randrange(1, 5)))


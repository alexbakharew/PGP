import numpy as np
import random
import os
det = 0
N = 9000
if not os.path.exists(".//tests"):
    os.mkdir(".//tests")
for i in range(1):
    with open(".//tests//input{}".format(i), "w") as test_in:
        matrix = np.random.rand(N, N)
        test_in.write(str(N) + "\n")
        for j in range(N):
            for k in range(N):
                test_in.write(str(matrix[j][k]) + " ")
            test_in.write("\n")
        det = np.linalg.det(matrix)

    with open(".//tests//output{}".format(i), "w") as test_out:
        test_out.write('%.10Lf\n'%det)




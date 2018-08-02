import numpy as np
import time
import  cProfile as profile
from numba import vectorize, cuda,jit


# @vectorize(['float32(float32, float32)'])
# @vectorize(['float32(float32, float32)'], target='cuda')
@jit(nopython = True, parallel = True, nogil = True)
def VectorAdd(a, b):
    c= a+b
    return c

def main():
    N = 320000000

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    start = time.time()
    for i in range(50):
        print(i)
        C = VectorAdd(A,B)
    vector_add_time = time.time() - start

    print ("C[:5] = " + str(C[:5]))
    print ("C[-5:] = " + str(C[-5:]))

    print ("VectorAdd took for % seconds" % vector_add_time)

# if __name__=='__main__':
# profile.run("main()",sort="time")
if __name__ == '__main__':
    main()
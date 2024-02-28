from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, vectorize
import time

@jit(nopython=True, nogil=True)
def jitTiler(arr, reps):
    n = len(arr)
    m = reps
    tiled = np.empty((n,m))
    for i in range(n):
        for j in range(m):
            tiled[i,j] = arr[j]

    return tiled

# @vectorize(nopython=True)
# def complexCalculator(z, c):
#     res = z**2+c
#     return res

# @vectorize(nopython=True)
# def complexMatrixMaker(x,y):
#     res = x + 1j * y
#     return res

# @vectorize(nopython=True)
# def matrixAddition(x : np.float64, y: np.bool_) -> np.float64:
#     return x + y


@jit(nopython=True, nogil=True)
def jitVectorized(width, height, T):
    # x = np.empty(width)
    # y = np.empty(height)

    # for i in range(width):
    #     x[i] = -2 + (i / (width - 1)) * 3

    # for j in range(height):
    #     y[j] = -1.5 + (j / (height - 1)) * 3


    x = -2 + (np.arange(width) / (width - 1)) * 3
    y = -1.5 + (np.arange(height) / (height - 1)) * 3

    
    x = jitTiler(x, len(y))
    y = jitTiler(y, len(x)).T


    c = x + 1j * y #complexMatrixMaker(x,y)
    
    z = np.zeros_like(c)
    output = np.zeros(c.shape)


    for _ in range(1000):
        z = z**2+c #complexCalculator(z, c)
        mask = np.abs(z) <= T
        output += mask#matrixAddition(output,mask)
        # output += mask
        if not np.any(mask):
            break

    return output



width = 5000
height = 5000
T = 2


# output = jitVectorized(100, 100, T)
start_time = time.time()
output = jitVectorized(width, height, T)
end_time = time.time()
print(f'Jit Vectorized took: {end_time-start_time}')

print(output.shape)

plt.imshow(output, cmap="hot", extent=[-2,1,-1.5,1.5],aspect='auto')
plt.colorbar()
plt.title(f"Mandelbrot {width}x{height}")
plt.show()
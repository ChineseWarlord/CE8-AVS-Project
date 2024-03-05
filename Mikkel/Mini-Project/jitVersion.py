import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time



@jit(nopython=True, nogil=True)
def jitApproach(width,height,T):


    x0 = -2 + (np.arange(width) / (width - 1)) * 3
    y0 = -1.5 + (np.arange(height) / (height - 1)) * 3

    T_squared = T*T

    output = np.empty((height,width), dtype=np.int32)
    iterations_max = 100
    for i in range(len(x0)):
        x0_i = x0[i]
        for j in range(len(y0)):
            x = 0.0
            y = 0.0
            y0_j = y0[j]
            iterations = 0
            while (iterations < iterations_max) and (x*x + y*y <= T_squared):
                x_temp = x*x - y*y + x0_i
                y = 2*x*y + y0_j
                x=x_temp
                iterations+=1
            output[j,i] = iterations

    
    return output


width = 5000
height = 5000
T = 2


start_time = time.time()
output=jitApproach(width, height, T)
end_time = time.time()
print(f'Jit took: {(end_time-start_time)}')
print(output.shape)


# plt.imshow(output, cmap="hot", extent=[-2,1,-1.5,1.5],aspect='auto')
# plt.colorbar()
# plt.title(f"Mandelbrot {width}x{height}")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time


def naiveApproach(width,height,T):
    x0 = np.linspace(-2,1,width)
    y0 = np.linspace(-1.5,1.5,height)

    output = np.zeros((height,width))
    for i in range(len(x0)):
        # print(f"Progress: {i+1}/{width}", end='\r')
        for j in range(len(y0)):
            x = 0.0
            y = 0.0
            iterations = 0
            iterations_max = 100
            while (x*x + y*y <= T*T) and (iterations < iterations_max):
                x_temp = x*x - y*y + x0[i]
                y = 2*x*y + y0[j]
                x=x_temp
                iterations+=1
            output[j,i] = iterations

    return output

def naiveApproachOptimized(width,height,T):
    x0 = np.linspace(-2,1,width)
    y0 = np.linspace(-1.5,1.5,height)

    output = np.empty((height,width), dtype=np.int32)

    T_squared = T*T

    for i in range(len(x0)):
        print(f"Progress: {i+1}/{width}", end='\r')
        x0_i = x0[i]
        for j in range(len(y0)):
            x = 0.0
            y = 0.0
            y0_j = y0[j]
            iterations = 0
            iterations_max = 100
            while (x*x + y*y <= T_squared) and (iterations < iterations_max):
                x_temp = x*x - y*y + x0_i
                y = 2*x*y + y0_j
                x=x_temp
                iterations+=1
            output[j,i] = iterations

    return output

width = 500
height = 500
T = 2


start_time = time.time()
for i in range(5):
    print(f"Progress: {i+1}/{5}", end='\r')
    output = naiveApproach(width, height, T)
end_time = time.time()
print(f'Naive took: {end_time-start_time}')

start_time = time.time()
for i in range(5):
    print(f"Progress: {i+1}/{5}", end='\r')
    output = naiveApproachOptimized(width, height, T)
end_time = time.time()
print(f'Naive took optimized: {end_time-start_time}')


# plt.imshow(output, cmap="hot", extent=[-2,1,-1.5,1.5],aspect='auto')
# plt.colorbar()
# plt.title(f"Mandelbrot {width}x{height}")
# plt.show()
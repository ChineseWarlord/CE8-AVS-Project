import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time



@jit
def jitApproach(width,height,T):
    x0 = np.empty(width)
    y0 = np.empty(height)

    for i in range(width):
        x0[i] = -2 + (i / (width - 1)) * 3

    for j in range(height):
        y0[j] = -1.5 + (j / (height - 1)) * 3

    output = np.zeros((height,width))
    for i in range(len(x0)):
        #print(f"Progress: {i+1}/{width}", end='\r')
        for j in range(len(y0)):
            x = 0.0
            y = 0.0
            iterations = 0
            iterations_max = 1000
            while (x*x + y*y <= T*T) and (iterations < iterations_max):
                x_temp = x*x - y*y + x0[i]
                y = 2*x*y + y0[j]
                x=x_temp
                iterations+=1
            output[j,i] = iterations

    
    return output


width = 1000
height = 1000
T = 2




start_time = time.time()
output = jitApproach(width, height, T)
end_time = time.time()
print(f'Jit took: {end_time-start_time}')


plt.imshow(output, cmap="hot", extent=[-2,1,-1.5,1.5],aspect='auto')
plt.colorbar()
plt.title(f"Mandelbrot {width}x{height}")
plt.show()

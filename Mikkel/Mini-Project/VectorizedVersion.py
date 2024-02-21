import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time



def vectorizedApproach(width, height, T):
    x = np.linspace(-2, 1, width)
    y = np.linspace(-1.5, 1.5, height)

    x, y = np.meshgrid(x, y)

    c = x + 1j * y
    z = np.zeros_like(c)
    output = np.zeros(c.shape)

    for _ in range(100):
        z = z**2 + c
        mask = np.abs(z) <= T
        output += mask
        z[~mask] = np.nan #z[mask]**2+c[mask]  # Update only the points still in the set

    return output



width = 5000
height = 5000
T = 2



start_time = time.time()
output = vectorizedApproach(width, height, T)
end_time = time.time()
print(f'Vectorized took: {end_time-start_time}')

print(output.shape)

plt.imshow(output, cmap="hot", extent=[-2,1,-1.5,1.5],aspect='auto')
plt.colorbar()
plt.title(f"Mandelbrot {width}x{height}")
plt.show()
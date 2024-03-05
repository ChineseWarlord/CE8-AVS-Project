"""
File: Mini-project.py
Author: Simon Loi Baks
Date: 06/03/2024
Description: This script contains multiple implementations of different Mandelbrot set algorithms.
"""

# %% [markdown]
# # Mini-Project Numerical Scientific Computing
#
# - Understand the mathematical algorithm and how it can be programmed.
# - Naive algorithm: Make a first implementation in Python using for-loops. You can use either lists or numpy arrays for data storage, but no vector operations.
# - Numpy vectorized algorithm: Instead of looping over every element, use Numpy vectorized operations.
# - Numba-optimized version(s).
# - Parallel version using multiprocessing.
#

# %% [markdown]
# # Importing libraries
#

# %%
# Importing libraries
import matplotlib.pyplot as plt
import multiprocessing as mp

# from tqdm.notebook import tqdm
from tqdm import tqdm
from numba import jit
import numpy as np
import cv2 as cv
import timeit
import time

# %% [markdown]
# # Naive algorithm for computing Mandelbrot set
#

# %% [markdown]
# ##### Naive Mandelbrot Algorithm
#


# %%
def NaiveMandelbrot(width, height, img, params):
    # Pseudocode from wikipedia: https://en.wikipedia.org/wiki/Mandelbrot_set
    min_real, max_real, min_imaginary, max_imaginary, max_iter, T = params

    # Iterate through each pixel coordinate
    for Px in tqdm(range(width), total=width, desc="Computing Mandelbrot Set"):
        for Py in range(height):
            # Map pixel coordinates to real and imaginary parts of c
            c_Real = min_real + (max_real - min_real) * (Px / width)
            c_Imgy = min_imaginary + (max_imaginary - min_imaginary) * (Py / height)

            # Initialize z as 0
            z_Real = 0
            z_Imgy = 0

            # Initialize iterations
            iter = 0

            # Calculate the Mandelbrot iteration until the threshold
            while iter < max_iter and z_Real * z_Real + z_Imgy * z_Imgy <= T * 2:
                # Calculate the next iteration of the real and imginary part of z
                next_z_Real = z_Real * z_Real - z_Imgy * z_Imgy + c_Real
                next_z_Imgy = 2 * z_Real * z_Imgy + c_Imgy

                # Assign the calculated parts to z as the new starting point
                z_Real = next_z_Real
                z_Imgy = next_z_Imgy

                # Increase iteration
                iter += 1

            # Checking if i(c) is in the Mandelbrot set
            if iter == max_iter:
                # In the Mandelbrot set
                img[Py, Px] = 255
            else:
                img[Py, Px] = iter

    return img


# %% [markdown]
# # Optimizing the inner while loop
#

# %% [markdown]
# ##### Optimized Mandelbrot Algorithm+
#


# %%
def NaiveMandelbrotOptimized(width, height, img, params):
    # Pseudocode from wikipedia: https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
    min_real, max_real, min_imaginary, max_imaginary, max_iter, T = params

    # Iterate through each pixel coordinate
    for Px in tqdm(range(width), total=width, desc="Computing Mandelbrot Set"):
        for Py in range(height):
            # Map pixel coordinates to real and imaginary parts of c
            c_Real = min_real + (max_real - min_real) * (Px / width)
            c_Imgy = min_imaginary + (max_imaginary - min_imaginary) * (Py / height)

            # Initialize z as 0 and introduce new variable w
            z_Real = 0
            z_Imgy = 0
            w = 0

            # Initialize iterations
            iter = 0

            # Calculate the Mandelbrot iteration until the threshold
            # Optimizing multiplication computations from naive approach
            while iter < max_iter and z_Real + z_Imgy <= T * 2:
                # Calculate the next iteration of the real and imginary part of z
                next_z_Real = z_Real - z_Imgy + c_Real
                next_z_Imgy = w - z_Real - z_Imgy + c_Imgy

                # Assign the calculated parts to z as the new starting point
                z_Real = next_z_Real * next_z_Real
                z_Imgy = next_z_Imgy * next_z_Imgy
                w = (next_z_Real + next_z_Imgy) * (next_z_Real + next_z_Imgy)

                # Increase iteration
                iter += 1

            # Checking if i(c) is in the Mandelbrot set
            if iter == max_iter:
                # In the Mandelbrot set
                img[Py, Px] = 255
            else:
                img[Py, Px] = iter

    return img


# %% [markdown]
# ##### Further Optimized Mandelbrot Algorithm++
#


# %%
def NaiveMandelbrotOptimizedPlus(width, height, img, params):
    # Pseudocode from wikipedia: https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
    min_real, max_real, min_imaginary, max_imaginary, max_iter, T = params

    # Iterate through each pixel coordinate
    for Px in tqdm(range(width), total=width, desc="Computing Mandelbrot Set"):
        for Py in range(height):
            # Map pixel coordinates to real and imaginary parts of c
            c_Real = min_real + (max_real - min_real) * (Px / width)
            c_Imgy = min_imaginary + (max_imaginary - min_imaginary) * (Py / height)

            # Initialize z as 0
            z_Real = 0
            z_Imgy = 0

            next_z_Real = 0
            next_z_Imgy = 0

            # Initialize iterations
            iter = 0

            # Calculate the Mandelbrot iteration until the threshold
            # Optimizing multiplication computations from naive approach
            while iter < max_iter and z_Real + z_Imgy <= T * 2:
                # Calculate the next iteration of the real and imginary part of z
                next_z_Imgy = 2 * next_z_Real * next_z_Imgy + c_Imgy
                next_z_Real = z_Real - z_Imgy + c_Real

                # Assign the calculated parts to z as the new starting point
                z_Real = next_z_Real * next_z_Real
                z_Imgy = next_z_Imgy * next_z_Imgy

                # Increase iteration
                iter += 1

            # Checking if i(c) is in the Mandelbrot set
            if iter == max_iter:
                # In the Mandelbrot set
                img[Py, Px] = 255
            else:
                img[Py, Px] = iter

    return img


# %% [markdown]
# # Mandelbrot algorithm using Numpy vectorization
#

# %% [markdown]
# Performing vectorization on the naive Mandelbrot Algorithm implementation
#


# %%
def VectorizedMandelbrot(width, height, img, params):
    min_real, max_real, min_imaginary, max_imaginary, max_iter, T = params

    # Creating np arrays for width/height pixels
    Px = np.linspace(min_real, max_real, width)
    Py = np.linspace(min_imaginary, max_imaginary, height)
    c = Px[np.newaxis, :] + 1j * Py[:, np.newaxis]

    # Initialize z as 0's, iter, and mask arrays
    z = np.zeros_like(c, dtype=np.complex128)
    iter = np.zeros(c.shape, dtype=np.int32)
    mask = np.full(c.shape, True, dtype=bool)

    # Using logical mask to check for divergence or escaped points
    t1 = time.time()
    for i in tqdm(range(int(max_iter)), total=int(max_iter), desc="Computing Mandelbrot Set"):
        z[mask] = z[mask] * z[mask] + c[mask]
        mask = np.logical_and(mask, abs(z) < T)
        iter[mask] = i

    img[:, :] = iter
    print(f"Execution time: {time.time() - t1}s")

    return img


# %% [markdown]
# # Naive Mandelbrot Algorithm using Numba jit
#


# %%
@jit(nopython=True, fastmath=True, cache=True)
def NaiveMandelbrotNumba(width, height, img, params):
    # Pseudocode from wikipedia: https://en.wikipedia.org/wiki/Mandelbrot_set
    min_real, max_real, min_imaginary, max_imaginary, max_iter, T = params

    # Iterate through each pixel coordinate
    for Px in range(width):
        for Py in range(height):
            # Map pixel coordinates to real and imaginary parts of c
            c_Real = min_real + (max_real - min_real) * (Px / width)
            c_Imgy = min_imaginary + (max_imaginary - min_imaginary) * (Py / height)

            # Initialize z as 0
            z_Real = 0
            z_Imgy = 0

            # Initialize iterations
            iter = 0

            # Calculate the Mandelbrot iteration until the threshold
            while z_Real**2 + z_Imgy**2 <= T * 2 and iter < max_iter:
                # Calculate the next iteration of the real and imginary part of z
                next_z_Real = z_Real**2 - z_Imgy**2 + c_Real
                next_z_Imgy = 2 * z_Real * z_Imgy + c_Imgy

                # Assign the calculated parts to z as the new starting point
                z_Real = next_z_Real
                z_Imgy = next_z_Imgy

                # Increase iteration
                iter += 1

            # Checking if i(c) is in the Mandelbrot set
            if iter == max_iter:
                # In the Mandelbrot set
                img[Py, Px] = 255
            else:
                img[Py, Px] = iter

    return img


# %% [markdown]
# # Parallel Mandelbrot Algorithm
#


# %%
def MandelbrotChunk(chunk_start, chunk_end, width, height, params):
    min_real, max_real, min_imaginary, max_imaginary, max_iter, T, *_ = params

    # Initialize an empty array to store the chunk
    chunk = np.zeros((chunk_end - chunk_start, width), dtype=np.uint8)

    # Iterating through a chunk of the image
    for Px in range(width):
        for Py in range(chunk_start, chunk_end):
            # Map pixel coordinates to real and imaginary parts of c
            c_Real = min_real + (max_real - min_real) * (Px / width)
            c_Imgy = min_imaginary + (max_imaginary - min_imaginary) * (Py / height)

            # Initialize z as 0
            z_Real = 0
            z_Imgy = 0

            # Initialize iterations
            iter = 0

            # Calculate the Mandelbrot iteration until the threshold
            while z_Real * z_Real + z_Imgy * z_Imgy <= T * 2 and iter < max_iter:
                # Calculate the next iteration of the real and imginary part of z
                next_z_Real = z_Real * z_Real - z_Imgy * z_Imgy + c_Real
                next_z_Imgy = 2 * z_Real * z_Imgy + c_Imgy

                # Assign the calculated parts to z as the new starting point
                z_Real = next_z_Real
                z_Imgy = next_z_Imgy

                # Increase iteration
                iter += 1

            # Checking if i(c) is in the Mandelbrot set
            if iter == max_iter:
                # In the Mandelbrot set
                color = 255
                chunk[Py - chunk_start, Px] = color
            else:
                chunk[Py - chunk_start, Px] = iter

    return chunk


def chunking(check, num_processes, height):
    if check == "equal":
        # Dividing image in equal chunks based on number of processes
        chunk_size = height // num_processes
        chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes)]

        # If the last chunk size != height, we add the remainder of pixels to the last chunk
        if chunks[-1][1] != height:
            chunks[-1] = (chunks[-1][0], height)

        return chunks

    elif check == "dynamic":
        # Using dynamic chunking based on number of processed and image size
        avg_chunk_size = height / num_processes
        remaining_height = height % num_processes
        chunks = []
        start = 0

        # Check if remaining height is less than numnber of processes
        # if yes, increase chunk size for making even chunks
        for i in range(num_processes):
            chunk_size = int(avg_chunk_size)
            if i < remaining_height:
                chunk_size += 1
            end = start + chunk_size
            chunks.append((start, end))
            start = end

        return chunks
    else:
        print("Error! *Wrong parameter*")


def ParallelMandelbrot(width, height, img, params):
    *_, check, P = params

    # Create process pool and apply chunking scheme
    pool = mp.Pool(processes=P)
    chunks = chunking(check, P, height)
    print("type:", check)
    print("processes:", P)
    print("chunks", chunks)

    # Assigning a chunk to each process
    results = []
    t1 = time.time()
    for chunk_start, chunk_end in chunks:
        results.append(pool.apply_async(MandelbrotChunk, (chunk_start, chunk_end, width, height, params)))

    # Combine each image chunk to create full image
    for result, (chunk_start, chunk_end) in zip(results, chunks):
        img[chunk_start:chunk_end] = result.get()

    # Close and wait for process termination -> Good practice!
    pool.close()
    pool.join()

    t2 = time.time()
    t = t2 - t1
    print("time:", t)

    return img, t


def plot_results(P_values, time_values, chunk):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Performance Analysis of Parallel Mandelbrot Algorithm\nChunking: {chunk}", fontsize=16)

    axs[0].set_title(f"Execution Time")
    axs[0].plot(P_values, time_values, marker="o", color="b")
    axs[0].set_xlabel("Number of Processes")
    axs[0].set_ylabel("Execution Time [s]")

    speedup = [time_values[0] / t for t in time_values]

    axs[1].set_title(f"Speedup")
    axs[1].plot(P_values, speedup, marker="o", color="r")
    axs[1].set_xlabel("Number of Processes")
    axs[1].set_ylabel("Speedup")

    plt.tight_layout()
    plt.show()


# %% [markdown]
# # Displaying the Mandelbrot set in the complex plane
#

# %% [markdown]
# ##### Function to display Mandelbrot
#


# %%
def displayMandelbrot(img, params, title, cmap=None):
    min_real, max_real, min_imaginary, max_imaginary, *_ = params
    plt.imshow(img, cmap=cmap, extent=(min_real, max_real, min_imaginary, max_imaginary))
    plt.xlabel("Real(c)")
    plt.ylabel("Imaginary(c)")
    plt.title("Mandelbrot Set\n- " + f"{title}")
    plt.show()


# %% [markdown]
# # Defining image and complex variables
#

# %%
# Defining image size
width = 5000
height = 5000
img = np.zeros((height, width), dtype=np.uint8)

# Defining max iterations
max_iter = 100.0

# Defining threshold
T = 2.0

# Defining the range of values for
# the real and imaginary parts of c
min_real = -2.0
max_real = 1.0
min_imaginary = -1.5
max_imaginary = 1.5
params = [min_real, max_real, min_imaginary, max_imaginary, max_iter, T]

# %% [markdown]
# ### Running algorithms
#

# %%
if __name__ == "__main__":
    # Naive Mandelbrot Algorithm
    print(f"\n- Naive Mandelbrot Algorithm -")
    t = time.time()
    img_mandel = NaiveMandelbrot(width, height, img, params)
    print(f"Execution time: {time.time() - t}")
    displayMandelbrot(img_mandel, params, "Naive", "hot")

    # Optimized Mandelbrot Algorithm+
    print(f"\n- Optimized Naive Mandelbrot Algorithm+ -")
    t = time.time()
    img_mandel = NaiveMandelbrotOptimized(width, height, img, params)
    print(f"Execution time: {time.time() - t}")
    displayMandelbrot(img_mandel, params, "Naive Optimized+", "hot")

    # Optimized Mandelbrot Algorithm++
    print(f"\n- Optimized Naive Mandelbrot Algorithm++ -")
    t = time.time()
    img_mandel = NaiveMandelbrotOptimizedPlus(width, height, img, params)
    print(f"Execution time: {time.time() - t}")
    displayMandelbrot(img_mandel, params, "Naive optimized++", "hot")

    # Vectorized Mandelbrot Algorithm
    print(f"\n- Vectorized Mandelbrot Algorithm -")
    t = time.time()
    img_mandel = VectorizedMandelbrot(width, height, img, params)
    print(f"Execution time: {time.time() - t}")
    displayMandelbrot(img_mandel, params, "Vectorized", "hot")

    # Numba jit Optimized Mandelbrot Algorithm
    print(f"\n- Numba Mandelbrot Algorithm -")
    t = time.time()
    img_mandel1 = NaiveMandelbrotNumba(width, height, img, params)
    print(f"Execution time: {time.time() - t}")
    displayMandelbrot(img_mandel1, params, "Numba", "hot")

    # Parallel Mandelbrot Algorithm
    processes = mp.cpu_count()
    P_values = range(1, processes + 1)
    times1 = []
    times2 = []
    for i in range(processes):
        params = [min_real, max_real, min_imaginary, max_imaginary, max_iter, T, "equal", i + 1]
        img1, t = ParallelMandelbrot(width, height, img, params)
        times1.append(t)

        params = [min_real, max_real, min_imaginary, max_imaginary, max_iter, T, "dynamic", i + 1]
        img2, t = ParallelMandelbrot(width, height, img, params)
        times2.append(t)

    print(f"\nChunking: Equal\nTotal Time: {np.array(times1).sum()}")
    print(f"\nChunking: Dynamic\nTotal Time: {np.array(times2).sum()}")

    displayMandelbrot(img1, params, "Parallel\n- equal chunking", "hot")
    displayMandelbrot(img2, params, "Parallel\n- dynamic chunking", "hot")

    plot_results(P_values, times1, "equal")
    plot_results(P_values, times2, "dynamic")

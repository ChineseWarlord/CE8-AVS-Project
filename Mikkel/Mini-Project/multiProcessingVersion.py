
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time
import multiprocessing as mp


def naiveApproachOptimized(chunk, y0 ,T):
    result = []
    T_squared = T*T
    iterations_max = 100

    for x0_i in chunk:
        chunk_result = []
        for y0_j in y0:
            x = 0.0
            y = 0.0
            iterations = 0
            
            while (iterations < iterations_max) and (x*x + y*y <= T_squared):
                x_temp = x*x - y*y + x0_i
                y = 2*x*y + y0_j
                x=x_temp
                iterations+=1
            chunk_result.append(iterations)
        result.append(chunk_result)

    return result


def chunking(x0, P):
    chunk_size = len(x0)//P
    chunks = []
    while(len(x0) >= 2*chunk_size):
        chunks.append(x0[:chunk_size])
        x0 = x0[chunk_size:]
    chunks.append(x0)

    return chunks






def runMulti(P, width, height, T):
    start_time = time.time()

    pool = mp.Pool(processes=P)

    x0 = np.linspace(-2,1,width)
    y0 = np.linspace(-1.5,1.5,height)

    # chunks = chunking(x0, P)
    chunks = chunking(x0, P*P)

    print("Processes: ", P)
    print("Chunks: ", len(chunks))

    results = []
    for chunk in chunks:
            results.append(pool.apply_async(naiveApproachOptimized, args=(chunk, y0, T)))


    output = []
    for result in results:
        for res in result.get():
            output.append(res)

    pool.close()
    pool.join()
    output = list(zip(*output))

    end_time = time.time()
    print(f'Multiprocessed, with jit took: {end_time-start_time} \n')

    # print(len(output), " x ",len(output[0]))
    # plt.imshow(output, cmap="hot", extent=[-2,1,-1.5,1.5],aspect='auto')
    # plt.colorbar()
    # plt.title(f"Mandelbrot {width}x{height}")
    # plt.show()

    return (end_time-start_time)



def plot_results(P_values, time_values): 
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Performance Analysis of Parallel Mandelbrot Algorithm", fontsize=16)

    axs[0].set_title(f"Execution Time")
    axs[0].plot(P_values, time_values, marker='o', color='b')
    axs[0].set_xlabel("Number of Processes")
    axs[0].set_ylabel("Execution Time [s]")

    speedup = [time_values[0] / t for t in time_values]
    axs[1].set_title(f"Speedup")
    axs[1].plot(P_values, speedup, marker='o', color='r')
    axs[1].set_xlabel("Number of Processes")
    axs[1].set_ylabel("Speedup")

    plt.tight_layout()
    plt.show()

if __name__ =="__main__":
    width = 5000
    height = 5000
    T = 2


    processers = mp.cpu_count()
    P_values = range(16, processers+1)
    times = []

    for i in P_values:
        times.append(runMulti(i, width, height, T))

    
    plot_results(P_values, times)
import cupy as cu
import numpy as np 
import time

N = 1024

################# CUPY #####################

out = np.zeros((N,N), dtype=np.float32)

a = cu.random.random((N,N), dtype=cu.float32)
b = cu.random.random((N,N), dtype=cu.float32)
tic = time.perf_counter()
c = a @ b
toc = time.perf_counter()
out = c.get()


print(f"cupy matrix multuply:  {toc - tic:0.4f} seconds")
##############################################


################# NUMPY #####################

out = np.zeros((N,N))

a = np.random.random((N,N))
b = np.random.random((N,N))
tic = time.perf_counter()
c = a @ b
toc = time.perf_counter()
out = c


print(f"numpy matrix multuply:  {toc - tic:0.4f} seconds")
##############################################
import cupy as cp

add_kernel = cp.RawKernel(r'''
extern "C" __global__
void my_add(const float* x1, const float* x2, float* y) {
    int idx= (blockDim.x * blockIdx.x + threadIdx.x)*gridDim.y*blockDim.y+blockDim.y * blockIdx.y + threadIdx.y;
    //int idx = blockDim.x * threadIdx.y + threadIdx.x;
    //int idx = threadIdx.x;
    y[idx] = x1[idx] + x2[idx];
}
''', 'my_add')

n1 = 1024
n2 = 4
N = n1*n2

x1 = cp.arange(N*N, dtype=cp.float32).reshape((N,N))
x2 = cp.arange(N*N, dtype=cp.float32).reshape((N,N))

y = cp.zeros((N, N), dtype=cp.float32)-1

import time

start_time = time.perf_counter()
for i in range(0, 1000):
    add_kernel((n1,n1), (n2,n2), (x1, x2, y))  # grid, block and arguments
    #print(i)
end_time = time.perf_counter()
print(end_time-start_time)
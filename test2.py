import cupy as cp

culc_gravity = cp.RawKernel(r'''
extern "C" __global__
void culc_gravity(float* mas, float* vel_x, float* vel_y, float* acc_x, float* acc_y, const float dt, const int N, const float dp) {
    int idx= (blockDim.x * blockIdx.x + threadIdx.x)*gridDim.y*blockDim.y+blockDim.y * blockIdx.y + threadIdx.y;
    
    int idx_y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_x = blockIdx.y * blockDim.y + threadIdx.y;

    float acc_x_n = 0.0f;
    float acc_y_n = 0.0f;
    for(int i = 0; i<N; i++){
        if(i==idx_y) {continue;}
        for(int j = 0; j<N; j++){ 
            if(j==idx_x) {continue;}
            float dx = (idx_x-j)*dp;
            float dy = (idx_y-i)*dp;
            int index = j + i*N;
            float d = sqrt(dx*dx + dy*dy);
            float acc = 1.0f * mas[index]/(d*d*d);
            acc_x_n += acc*dx;
            acc_y_n += acc*dy;
        }
    }

    acc_x[idx] = acc_x_n;
    acc_y[idx] = acc_y_n;
    vel_x[idx] = vel_x[idx] + acc_x[idx] * dt;
    vel_y[idx] = vel_y[idx] + acc_y[idx] * dt;
    int k_x = idx_x - int(vel_x[idx]/dp);
    int k_y = idx_y - int(vel_y[idx]/dp);
    if(k_y<0 || k_y>=N || k_x<0 || k_x>=N){
        mas[idx] = 0;
        return;
    }
    int k = k_x + k_y*N;
    mas[idx] += mas[k];
}
''', 'culc_gravity')


n1 = 6
n2 = 2
N = n1*n2


dt = 0.01
dp = 0.01
mas   = cp.zeros((N,N), dtype=cp.float32) + 1
vel_x = cp.zeros((N,N), dtype=cp.float32)
vel_y = cp.zeros((N,N), dtype=cp.float32)
acc_x = cp.zeros((N,N), dtype=cp.float32)
acc_y = cp.zeros((N,N), dtype=cp.float32)

import time

start_time = time.perf_counter()

for i in range(0,100):
    culc_gravity((n1,n1), (n2,n2), (mas, vel_x, vel_y, acc_x, acc_y, dt, N, dp))

end_time = time.perf_counter()
print(mas)
print(acc_x)
print(end_time-start_time)
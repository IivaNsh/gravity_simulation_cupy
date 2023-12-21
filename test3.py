import cupy as cp
import numpy as np

culc_gravity = cp.RawKernel(r'''
extern "C" __global__
void culc_gravity(const float* mas_in, float* mas_out, const float* vel_x_in, const float* vel_y_in,float* vel_x_out, float* vel_y_out, float* acc_x_out, float* acc_y_out, float dt, int N, float scale) {
    int idx= (blockDim.x * blockIdx.x + threadIdx.x)*gridDim.y*blockDim.y+blockDim.y * blockIdx.y + threadIdx.y;
    
    int idx_y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_x = blockIdx.y * blockDim.y + threadIdx.y;

    float acc_x_n = 0.0f;
    float acc_y_n = 0.0f;
    for(int i = 0; i<N; i++){
        for(int j = 0; j<N; j++){ 
            if(j==idx_x && i==idx_y) {continue;}
            
            float dx = (float(j)-float(idx_x))*scale;
            float dy = (float(i)-float(idx_y))*scale;
            int index = j + i*N;
            float d = max(sqrtf(dx*dx + dy*dy), 0.001f);
            float acc = mas_in[index]/(d*d);//*(pow(d,2)-1.f)/(2.f*exp(0.5f*pow(d,2)));
            acc_x_n += acc*dx/d;
            acc_y_n += acc*dy/d;
        }
    }

    acc_x_out[idx] = acc_x_n;
    acc_y_out[idx] = acc_y_n;
    vel_x_out[idx] = vel_x_in[idx] + acc_x_n/mas_in[idx] * dt;
    vel_y_out[idx] = vel_y_in[idx] + acc_y_n/mas_in[idx] * dt;
    
    float xx = float(idx_x) - vel_x_out[idx]*dt;
    float yy = float(idx_y) - vel_y_out[idx]*dt;
    float w_x = xx - floor(xx);
    float w_y = yy - floor(yy);
    int k1_x = int(floor(xx));
    int k1_y = int(floor(yy));
    int k2_x = int(ceil(xx));
    int k2_y = int(floor(yy));
    int k3_x = int(ceil(xx));
    int k3_y = int(ceil(yy));
    int k4_x = int(floor(xx));
    int k4_y = int(ceil(yy));
    
    
    if(k1_y<0 || k1_y>=N || k1_x<0 || k1_x>=N || 
    k2_y<0 || k2_y>=N || k2_x<0 || k2_x>=N || 
    k3_y<0 || k3_y>=N || k3_x<0 || k3_x>=N ||
    k4_y<0 || k4_y>=N || k4_x<0 || k4_x>=N){
        mas_out[idx] = 0.0;
        return;
    }
    int k1 = k1_x + k1_y*N;
    int k2 = k2_x + k2_y*N;
    int k3 = k3_x + k3_y*N;
    int k4 = k4_x + k4_y*N;
    
    float v1 = mas_in[k1]+(mas_in[k4]-mas_in[k1])*w_x;
    float v2 = mas_in[k2]+(mas_in[k3]-mas_in[k2])*w_x;
    float m = v2+(v1-v2)*w_y;
    
    mas_out[idx] = mas_out[idx] + m*dt;
}
''', 'culc_gravity')


n1 = 16
n2 = 16
N = n1*n2


dt = 0.03
scale = 0.1
mas_in   = np.zeros((N,N), dtype=cp.float32)
for i in range(0,N):
    for j in range(0,N):
        mas_in[i,j] += 1/(np.sqrt((j-N/2-N/4)**2 + (i-N/2)**2)+1)
        mas_in[i,j] += 1/(np.sqrt((j-N/2+N/4)**2 + (i-N/2)**2)+1)
mas_in = cp.array(mas_in*50)

mas_out = cp.zeros((N,N), dtype=cp.float32)

vel_x_in = cp.zeros((N,N), dtype=cp.float32)
vel_y_in = cp.zeros((N,N), dtype=cp.float32)
for i in range(0,N):
    for j in range(0,N):
        a = np.arctan2(i-N/2,j-N/4)
        vel_x_in[i,j] = np.cos(a)*1000
        vel_y_in[i,j] = np.sin(a)*1000

vel_x_out = cp.zeros((N,N), dtype=cp.float32)
vel_y_out = cp.zeros((N,N), dtype=cp.float32)

acc_x = cp.zeros((N,N), dtype=cp.float32)
acc_y = cp.zeros((N,N), dtype=cp.float32)


import matplotlib.pyplot as plt
import matplotlib.animation as animation

T = 100

plt.style.use('dark_background')
fig = plt.figure(facecolor=(0.035, 0.035, 0.036))
ax = fig.subplots()


max_v = 5
min_v = 0

#culc_gravity((n1,n1), (n2,n2), (mas_in, mas_out, vel_x_in, vel_y_in, vel_x_out, vel_y_out, acc_x, acc_y, cp.float32(dt), N, cp.float32(scale)))
#ax.imshow(mas_out.get(), vmin=min_v, vmax=max_v, cmap="inferno")
#print(mas_out)

ims = []
for i in range(0,T):
    culc_gravity((n1,n1), (n2,n2), (mas_in, mas_out, vel_x_in, vel_y_in, vel_x_out, vel_y_out, acc_x, acc_y, cp.float32(dt), N, cp.float32(scale)))
    im = ax.imshow(mas_out.get(), animated=True, vmin=min_v, vmax=max_v, cmap="inferno")
    if i == 0:
        ax.imshow(mas_out.get(), vmin=min_v, vmax=max_v, cmap="inferno")  # show an initial one first
        print(mas_out)
    ims.append([im])
    mas_in = mas_out
    vel_x_in = vel_x_out
    vel_y_in = vel_y_out
    print(i)


ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
plt.show()
import cupy as cp
import numpy as np

culc_gravity = cp.RawKernel(r'''
extern "C" __global__
void culc_gravity(float* pos_x_in, float* pos_y_in, float* pos_x_out, float* pos_y_out, const float* mas, const float* vel_x_in, const float* vel_y_in,float* vel_x_out, float* vel_y_out, float* acc_x_out, float* acc_y_out, float dt, int N, float scale) {
    int idx= (blockDim.x * blockIdx.x + threadIdx.x)*gridDim.y*blockDim.y+blockDim.y * blockIdx.y + threadIdx.y;

    float acc_x_n = 0.0f;
    float acc_y_n = 0.0f;
    for(int i = 0; i<N; i++){
        if(i == idx) {continue;}
        float dx = pos_x_in[i]-pos_x_in[idx];
        float dy = pos_y_in[i]-pos_y_in[idx];
        float d = max(sqrtf(dx*dx + dy*dy), 0.01f);
        float acc = mas[i]/(d*d);
        acc_x_n += acc*dx/d;
        acc_y_n += acc*dy/d;
    }

    acc_x_out[idx] = acc_x_n;
    acc_y_out[idx] = acc_y_n;
    vel_x_out[idx] = vel_x_in[idx] + acc_x_out[idx] * dt;
    vel_y_out[idx] = vel_y_in[idx] + acc_y_out[idx] * dt;
    pos_x_out[idx] = pos_x_in[idx] + vel_x_out[idx] * dt;
    pos_y_out[idx] = pos_y_in[idx] + vel_y_out[idx] * dt;
}
''', 'culc_gravity')


n1 = 8
n2 = 8
N = n1*n2


dt = 0.01
scale = 0.1

pos_x_in = (cp.random.random((N,N), dtype = cp.float32)-0.5)*20
pos_y_in = (cp.random.random((N,N), dtype = cp.float32)-0.5)*20
pos_x_out = cp.zeros((N,N), dtype = cp.float32)
pos_y_out = cp.zeros((N,N), dtype = cp.float32)

vel_x_in = cp.zeros((N,N), dtype=cp.float32)
vel_y_in = cp.zeros((N,N), dtype=cp.float32)
for i in range(0, N):
    for j in range(0, N):
        a = np.arctan2(pos_x_in[i,j], pos_y_in[i,j])
        r = np.sqrt(pos_x_in[i,j]**2+pos_y_in[i,j]**2)
        vel_x_in[i,j] += cp.cos(a)*1
        vel_y_in[i,j] -= cp.sin(a)*1

vel_x_out = cp.zeros((N,N), dtype=cp.float32)
vel_y_out = cp.zeros((N,N), dtype=cp.float32)

acc_x = cp.zeros((N,N), dtype=cp.float32)
acc_y = cp.zeros((N,N), dtype=cp.float32)


mas = cp.zeros((N,N), dtype=cp.float32)
for i in range(0, N):
    for j in range(0, N):
        a = np.arctan2(pos_x_in[i,j]-10, pos_y_in[i,j]-10)
        r = np.sqrt(pos_x_in[i,j]**2+pos_y_in[i,j]**2)
        mas[i,j] = 1/((r+1)*(r+1))




import matplotlib.pyplot as plt
import matplotlib.animation as animation

T = 100

plt.style.use('dark_background')
fig = plt.figure()
ax = fig.subplots()
ax.set_aspect('equal', 'box')
ax.set_xlim([-16,16])
ax.set_ylim([-16,16])
ax.axis("off")


sizes = np.ones(N*N)*0.2

ptss_X = [pos_x_in.get().reshape(1,N*N)[0]]
ptss_Y = [pos_y_in.get().reshape(1,N*N)[0]]
col = [cp.sqrt(vel_x_out*vel_x_out+vel_y_out*vel_y_out).get().reshape(1,N*N)[0]]

PLOT = ax.scatter(ptss_X[0],ptss_Y[0],s=sizes, c=col[0], vmin=0, vmax=20, cmap='inferno')


for i in range(0,T):
    culc_gravity((n1,n1), (n2,n2), (pos_x_in, pos_y_in, pos_x_out, pos_y_out, mas, vel_x_in, vel_y_in, vel_x_out, vel_y_out, acc_x, acc_y, cp.float32(dt), N*N, cp.float32(scale)))

    ptss_X.append(pos_x_out.get().reshape(1,N*N)[0])
    ptss_Y.append(pos_y_out.get().reshape(1,N*N)[0])
    col.append(cp.sqrt(vel_x_out*vel_x_out+vel_y_out*vel_y_out).get().reshape(1,N*N)[0])

    pos_x_in = pos_x_out
    pos_y_in = pos_y_out
    vel_x_in = vel_x_out
    vel_y_in = vel_y_out
    print(i)


def update(frame):
    PLOT.set_offsets(np.array([ptss_X[frame],ptss_Y[frame]]).T)
    PLOT.set_array(col[frame])
    PLOT.set_sizes(sizes)
    return PLOT


ani = animation.FuncAnimation(fig=fig, func=update, frames=T, interval=50)

dpi = 300
writer = animation.writers['ffmpeg'](fps=35)
ani.save('test.mp4',writer=writer,dpi=dpi)



plt.show()
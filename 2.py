import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




def f3(x): return 1/x**2
def f(x): return 1/(x**2+1)**2
def f2(x): return (x**3-1)/(2*np.exp(0.5*x**3))

N = 100
points_pos = np.random.random((N, 2))-0.5 
points_vel = np.random.random((N, 2))-0.5 
points_acc = np.random.random((N, 2))-0.5 
points_mas = ((np.zeros((N, 2))+1).T*np.random.random(N)).T


mat = np.zeros((N,N,2))
def culc_matrix(points_pos):
    global mat
    for i in range(0,N):
        for j in range(i+1,N):
            vec = points_pos[j] - points_pos[i]
            r = np.abs(np.linalg.norm(vec))
            force = 5*f(r)*vec/r
            mat[i,j] = force
            mat[j,i] = -force
    return mat

dt = 0.01

def culc_next(points_pos, points_vel, points_acc, points_mas):
    mat = culc_matrix(points_pos)
    points_acc = (points_mas * mat).sum(axis=1)
    points_vel += points_acc*dt
    points_pos += points_vel*dt
    return (points_pos, points_vel, points_acc, points_mas)












plt.style.use('dark_background')
fig = plt.figure(facecolor=(0.035, 0.035, 0.036))
ax = fig.subplots()
ax.set_facecolor(color="#09090a")
ax.axis('equal')
ax.grid(alpha=0.5)
ax.axvline(x=0, c="bisque")
ax.axhline(y=0, c="bisque")

points_obj = ax.plot(points_pos[:,0],points_pos[:,1], "o", color = "coral")[0]


def update(frame):
    global points_pos, points_vel, points_acc, points_mas
    points_pos, points_vel, points_acc, points_mas = culc_next(points_pos, points_vel, points_acc, points_mas)
    points_obj.set_xdata(points_pos[:,0])
    points_obj.set_ydata(points_pos[:,1])
    return points_obj
    print(frame)


ani = animation.FuncAnimation(fig=fig, func=update, frames=1000, interval=1)



plt.show()
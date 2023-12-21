


#y0 = [psi_0 psiT_0]
#y = [psi psiT]
#y' = [psiT psiTT]


import numpy as np

def dt_function(y, t, k, k2, k3):
    yt = np.zeros(2)
    yt[0] = y[1]
    yt[1] = -k*np.sin(y[0])+k2*y[1]+k3*y[0]
    return yt

R = 2
g = 9.81
k = g/R
k2 = -0.1
k3 = -1

y0 = [0.0, 10]
t = np.linspace(0, 100, 1001)


from scipy.integrate import odeint
y = odeint(dt_function, y0, t, args=(k,k2,k3))

Psi = y[:, 0]
PsiT = y[:, 1]
PsiTT = np.zeros(len(t))
for i in range(0, len(t)):
    PsiTT[i] = dt_function([y[i,0],y[i,1]], t, k,k2,k3)[1]

import matplotlib.pyplot as plt
plt.style.use('dark_background')

fig = plt.figure(facecolor=(0.035, 0.035, 0.036))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.set_facecolor(color="#09090a")
ax1.plot(t, Psi, 'tomato', label='Psi(t)',alpha=0.7)
ax1.plot(t, PsiT, 'springgreen', label='PsiT(t)',alpha=0.7)
ax1.plot(t, PsiTT, 'royalblue', label='PsiTT(t)',alpha=0.7)
ax1.legend(loc='best')
ax1.axis('equal')
ax1.grid(alpha=0.5)

ax2.set_facecolor(color="#09090a")
ax2.plot(Psi, PsiT, "springgreen")
ax2.axis('equal')
ax2.grid(alpha=0.5)

ax3.set_facecolor(color="#09090a")
ax3.plot(Psi, PsiTT, "royalblue")
ax3.axis('equal')
ax3.grid(alpha=0.5)

ax4.set_facecolor(color="#09090a")
ax4.plot(PsiT, PsiTT, "indigo")
ax4.axis('equal')
ax4.grid(alpha=0.5)





import matplotlib.animation as animation

fig1, ax = plt.subplots()
ax.axis('equal')
ax.grid(alpha=0.5)

xs = -R*np.cos(Psi+np.pi/2)
ys = -R*np.sin(Psi+np.pi/2)
ax.set(xlim=[-3, 3], ylim=[-3, 3])

line = ax.plot([0, xs[0]], [0, ys[0]], "-", color="orange")[0]
mass = ax.plot([xs[0]], [ys[0]],  "o", color="orange")[0]

def update(frame):
    
    line.set_xdata([0, xs[frame]])
    line.set_ydata([0, ys[frame]])
    
    mass.set_xdata(xs[frame])
    mass.set_ydata(ys[frame])
    return (line, mass)


ani = animation.FuncAnimation(fig=fig1, func=update, frames=1000, interval=20)



plt.show()
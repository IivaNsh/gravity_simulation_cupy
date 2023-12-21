import numpy as np

def f(x): return (x**3-1)/(2*np.exp(0.5*x**3))

N = 3

pos = np.random.random((N,2)) - 0.5
acc = np.random.random((N,2)) - 0.5
mas = ((np.zeros((N, 2))+1).T*np.random.random(N)).T
print(mas)
print(acc)
print()
print()

mat = np.zeros((N,N,2))
for i in range(0,N):
        for j in range(i+1,N):
            vec = pos[j] - pos[i]
            r = np.abs(np.linalg.norm(vec))
            force = f(r)*vec/r
            mat[i,j] = force
            mat[j,i] = -force

acc = (mas * mat).sum(axis=1)
print(acc)

#a  = np.arange(N*2).reshape((N,N,2))
#print(a)
#print()
#
##b = np.arange(2)+1
#b= np.array([[1,1],[2,2],[3,3]])
#print(b)
#print()
#
#prod = b * a
#print(prod)
#print()
#
#print(prod.sum(axis=1))
import numpy as np

a  = np.arange(9).reshape((3,3))
print(a)
print()

b = a.reshape(1,9)[0]
print(b)

#a  = np.arange(18).reshape((3,3,2))
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
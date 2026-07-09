import numpy as np

v1 = np.array([2,4,6])
v2 = np.array([1,3,5])

print("Addition:", v1 + v2)
print("Dot Product:", np.dot(v1,v2))

m1 = np.array([[1,2],[3,4]])
m2 = np.array([[5,6],[7,8]])

print("Matrix Multiplication:")
print(np.matmul(m1,m2))

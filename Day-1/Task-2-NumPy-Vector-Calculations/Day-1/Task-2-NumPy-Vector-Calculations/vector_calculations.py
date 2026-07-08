import numpy as np

# Create vectors
vector1 = np.array([2, 4, 6])
vector2 = np.array([1, 3, 5])

print("Vector 1:", vector1)
print("Vector 2:", vector2)

# Vector Addition
print("\nVector Addition:")
print(vector1 + vector2)

# Dot Product
print("\nDot Product:")
print(np.dot(vector1, vector2))

# Matrix Operations
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

print("\nMatrix Multiplication:")
print(np.matmul(matrix1, matrix2))

print("\nTranspose:")
print(matrix1.T)

# Eigenvalues and Eigenvectors
values, vectors = np.linalg.eig(matrix1)

print("\nEigenvalues:")
print(values)

print("\nEigenvectors:")
print(vectors)

import numpy as np

from Basics.Magic import magic

a = np.matrix('1 2;3 4;10 12')
print("Matrix:\n", a, "\n")
print("Size (row , column)", a.shape)
a = np.matrix('1 ;2; 3')
print("Vector:\n", a, "\n")

a = np.ones([2, 2])
print("Ones:\n", a, "\n")

a = np.eye(2)
print("Identity:\n", a, "\n")

a = np.random.random((3, 3))
print("Random:\n", a, "\n")

a = np.random.randn(1, 3)
print("Randn:\n", a, "\n")

# a = list(-6 + np.math.sqrt(10) * (np.random.randn(1, 10000)))
# print(np.shape(a))

# mplot.hist(a, 50)
# mplot.title("Histogram Example")
# mplot.xlabel("Range")
# mplot.ylabel("Count")
# mplot.show()

x = np.loadtxt("data/ex1data2.txt", delimiter=',')
y = np.arange(1, 7)
print("File data:")
print(y)
x = x[y, :]
x = x[:, [1, 2]]
print("File data vstack:")
x = np.vstack([x, [10, 11]])
print(x)

print("Matrix a:")
a = np.array([1, 3, 4])
print(a)
print("Matrix b:")
b = np.array([5, 3, 7])
print(b)
print("Vstacka ab:")
c = np.vstack([a, b])
print(c)
print("Hstack ab:")
c = np.hstack([a, b])
print(c)
print("To vector(flatten):")
to_vector_x = np.array(x).flatten()
print(to_vector_x)

a = np.array([[1, 2], [3, 4], [5, 6]], float)
print("A: \n", a)

c = np.array([[1, 1], [2, 2]], float)
print("C: \n", c)

print("A*C: \n")
print(np.dot(a, c))

print("A^3: \n")
print(np.power(a, 3))

print("A reciprocal: \n")
print(np.reciprocal(a))
print("A': \n")
print(np.transpose(a))
np.max(a)
print("A argmax: \n")
print(a.max(), a.argmax(axis=0))
print("A max column axis=1:")
print(a.max(axis=1))
print("A max row axis=0:")
print(a.max(axis=0))

print("Magic: ")
A = magic(3)
print(A)
r, c = np.nonzero(A >= 7)
print("Greater than 7 indices: ")
print(r, c)

print("Matrix a:")
a = np.array([[1, 15], [2, 0.5]])
print(a)
print("Sum a:")
print(np.sum(a))
print("Prod a:")
print(np.prod(a))
print("Floor a:")
print(np.floor(a))
print("Ceil a:")
print(np.ceil(a))
print("ElementWise Max function:")
rand1 = np.random.random([3, 3])
rand2 = np.random.random([3, 3])
print("Array 1:")
print(rand1)
print("Array 2:")
print(rand2)
print("ElementWise Max:")
print(np.maximum(rand1, rand2))

print("Magic A: ")
A = magic(9)

print("Sum A:")
print(A)
print(np.sum(A, axis=0))
print(np.sum(A, axis=1))

C = np.eye(9) * A
print("Diagonal sum:")
print(C)
print(np.sum(np.sum(C, 0)))
B = np.flipud(np.eye(9))
C = A * B
print("Diagonal sum:")
print(C)
print(np.sum(np.sum(C, 0)))

print("Magic A: ")
A = magic(3)
print(A)
inv_A = np.linalg.pinv(A)
print("inv_A: ")
print(inv_A)
print("A*inv_A: ")
idnt = np.dot(A, inv_A)
print(idnt)
print(np.round(np.abs(idnt)))

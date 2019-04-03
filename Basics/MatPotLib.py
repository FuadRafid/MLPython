import matplotlib.pyplot as plt
import numpy as np

from Basics.Magic import magic

t = np.linspace(0, 1, 100)
y1 = np.sin(2 * np.pi * 4 * t)
y2 = np.cos(2 * np.pi * 4 * t)
y3 = np.tan(t)
plt.title("waves")
plt.subplot(121)
plt.plot(t, y1)
plt.subplot(122)
plt.plot(t, y2, "r")
# plt.plot(t,y3)
plt.axis([.5, 1, -1, 1])
plt.xlabel('time')
plt.ylabel('value')
plt.legend(['sin', 'cos', 'tan'])
plt.savefig('foo.png')
plt.show()

A = magic(15)
plt.imshow(A, cmap='Greys')
plt.colorbar()
plt.show()
print(A[7][14])

A = np.random.random([15, 15])
plt.imshow(A, cmap='RdBu')
plt.colorbar()
plt.show()
print(A[7][14])

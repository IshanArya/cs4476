import numpy as np
import matplotlib.pyplot as plt

try:
    A = np.load("q3-input.npy")
except:
    quit()
plt.gray()
# part a
sorted_matrix = np.sort(A, None)[::-1]
np.save("q3-output-sorted", sorted_matrix)
plt.plot(sorted_matrix)
plt.savefig("q3a.jpg")
plt.show()

#part b
plt.hist(A.flatten(), bins=20)
plt.savefig("q3b.jpg")
plt.show()

#part c
X = A[50:,:50]
np.save("q3-output-x", X)
plt.imshow(X)
plt.savefig("q3c.jpg")
plt.show()

#part d
matrix_mean = np.mean(A)
Y = A - matrix_mean
np.save("q3-output-y", Y)
plt.imshow(Y)
plt.savefig("q3d.jpg")
plt.show()

# part e
Z = np.where(A > matrix_mean, 1., 0.)
Z = np.repeat(Z[:,:,np.newaxis], 3, axis=2)
Z[:,:,1:] = 0.
plt.imshow(Z)
plt.savefig("q3-output-z.png")
plt.show()
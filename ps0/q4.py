import numpy as np
import imageio
import matplotlib.pyplot as plt

im = imageio.imread("q4-input.png")

# part a
A = np.copy(im)
A[:,:,0] = im[:,:,1]
A[:,:,1] = im[:,:,0]
imageio.imwrite("q4-output-swapped.png", A)

# part b
B = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
B[:, :] = np.mean(A, axis=2)
imageio.imwrite("q4-output-grayscale.png", B)

# part c
C = 255 - im
imageio.imwrite("q4-output-negative.png", C)

# part d
D = B[:,::-1]
imageio.imwrite("q4-output-mirror.png", D)

# part e
E = (D + B)//2
imageio.imwrite("q4-output-average.png", E)

# part f
N = (np.random.rand(*B.shape) * 256).astype(np.uint32)
F = np.clip(N + B.astype(np.uint32), None, 255).astype(np.uint8)
np.save("q4-noise", N)
imageio.imwrite("q4-output-noise.png", F)

plt.gray()
fig, axs = plt.subplots(3,2)
axs[0, 0].imshow(A)
axs[0, 0].set_title("GRB Image")
axs[0, 1].imshow(B)
axs[0, 1].set_title("Grayscale Image")
axs[1, 0].imshow(C)
axs[1, 0].set_title("Negative Image")
axs[1, 1].imshow(D)
axs[1, 1].set_title("Flipped Image")
axs[2, 0].imshow(E)
axs[2, 0].set_title("Mirror Image")
axs[2, 1].imshow(F)
axs[2, 1].set_title("Noisy Image")
fig.tight_layout(pad=2.0)
plt.savefig("q4subplot.jpg")
plt.show()
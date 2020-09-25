import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.draw import circle_perimeter
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from imageio import imread, imwrite


def detectCircles(img, radius, use_gradient=False):
    gray_img = rgb2gray(img).astype(np.float32)
    edges = canny(gray_img, sigma=3)

    gradient_x, gradient_y = np.gradient(gray_img)

    thetas = np.arctan(gradient_y/gradient_x)

    rows, columns = edges.shape

    accum = np.zeros_like(gray_img)

    for i in range(rows):
        for j in range(columns):
            if edges[i][j]:
                if use_gradient:
                    theta = thetas[i][j]
                    a = round(i + radius * np.cos(theta))
                    b = round(j + radius * np.sin(theta))
                    if a >= 0 and b >= 0 and a < rows and b < columns:
                        accum[a][b] += 1
                    else:
                        a = round(i + radius * np.cos(theta + np.pi))
                        b = round(j + radius * np.sin(theta + np.pi))
                        if a >= 0 and b >= 0 and a < rows and b < columns:
                            accum[a][b] += 1
                else:
                    for theta in np.arange(0, 2 * np.pi, 0.01):
                        a = round(i + radius * np.cos(theta))
                        b = round(j + radius * np.sin(theta))
                        if a >= 0 and b >= 0 and a < rows and b < columns:
                            accum[a][b] += 1

    plt.imshow(accum)
    plt.show()

    return np.column_stack(np.where(accum >= 0.9 * np.amax(accum)))

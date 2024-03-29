
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# Edit KMeans.ipynb instead.
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv, hsv2rgb
from typing import Tuple

def quantize_rgb(img: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the k-means clusters for the input image in RGB space, and return
    an image where each pixel is replaced by the nearest cluster's average RGB
    value.

    Inputs:
        img: Input RGB image with shape H x W x 3 and dtype "uint8"
        k: The number of clusters to use

    Output:
        An RGB image with shape H x W x 3 and dtype "uint8"
    """
    w, h, d = img.shape
    img_array = np.reshape(img, (w * h, d))
    quantized_img = np.zeros_like(img)

    ##########################################################################
    # TODO: Perform k-means clustering and return an image where each pixel  #
    # is assigned the value of the nearest clusters RGB values.              #
    ##########################################################################

    kmeans = KMeans(n_clusters=k, random_state=101)
    labels = kmeans.fit_predict(img_array)

    codebook = kmeans.cluster_centers_

    label_idx = 0
    for i in range(w):
      for j in range(h):
        quantized_img[i][j] = codebook[labels[label_idx]]
        label_idx += 1



    ##########################################################################
    ##########################################################################

    return quantized_img

def quantize_hsv(img: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the k-means clusters for the input image in the hue dimension of the
    HSV space. Replace the hue values with the nearest cluster's hue value. Finally,
    convert the image back to RGB.

    Inputs:
        img: Input RGB image with shape H x W x 3 and dtype "uint8"
        k: The number of clusters to use

    Output:
        An RGB image with shape H x W x 3 and dtype "uint8"
    """
    hsv_img = rgb2hsv(img)
    w, h, d = hsv_img.shape
    img_array = np.reshape(hsv_img, (w * h, d))
    img_array = np.delete(img_array, (1, 2), 1)
    quantized_img = np.zeros_like(img)

    ##########################################################################
    # TODO: Convert the image to HSV. Perform k-means clustering in hue      #
    # space. Replace the hue values in the image with the cluster centers.   #
    # Convert the image back to RGB.                                         #
    ##########################################################################

    kmeans = KMeans(n_clusters=k, random_state=101)
    labels = kmeans.fit_predict(img_array)


    codebook = kmeans.cluster_centers_

    hsv_img[:,:,0] = codebook[labels].reshape(hsv_img[:,:,0].shape)

    quantized_img = (hsv2rgb(hsv_img) * 255).astype(np.uint8)

    ##########################################################################
    ##########################################################################

    return quantized_img

def compute_quantization_error(img: np.ndarray, quantized_img: np.ndarray) -> int:
    """
    Compute the sum of squared error between the two input images.

    Inputs:
        img: Input RGB image with shape H x W x 3 and dtype "uint8"
        quantized_img: Quantized RGB image with shape H x W x 3 and dtype "uint8"

    Output:

    """
    error = 0

    ##########################################################################
    # TODO: Compute the sum of squared error.                                #
    ##########################################################################

    error = np.sum(np.square(img.astype(np.uint32)-quantized_img.astype(np.uint32)))

    ##########################################################################
    ##########################################################################

    return error

def get_hue_histograms(img: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the histogram values two ways: equally spaced and clustered.

    Inputs:
        img: Input RGB image with shape H x W x 3 and dtype "uint8"
        k: The number of clusters to use

    Output:
        hist_equal: The values for an equally spaced histogram
        hist_clustered: The values for a histogram of the cluster assignments
    """
    hist_equal = np.zeros((k,), dtype=np.int64)
    hist_clustered = np.zeros((k,), dtype=np.int64)

    ##########################################################################
    # TODO: Convert the image to HSV. Calculate a k-bin histogram for the    #
    # hue dimension. Calculate the k-means clustering of the hue space.      #
    # Calculate the histogram values for the cluster assignments.            #
    ##########################################################################

    hsv_img = rgb2hsv(img)

    hues = hsv_img[:,:,0].reshape((-1, 1))

    hist_equal, _ = np.histogram(hues, bins=k)


    kmeans = KMeans(n_clusters=k, random_state=101)
    labels = kmeans.fit_predict(hues)

    _, hist_clustered = np.unique(labels, return_counts=True)

    ##########################################################################
    ##########################################################################

    return hist_equal, hist_clustered
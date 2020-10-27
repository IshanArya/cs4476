import glob
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from skimage.color import rgb2gray

from displaySIFTPatches import displaySIFTPatches
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from selectRegion import roipoly


framesdir = "/Users/ishanarya/Desktop/school/cv/ps/ps4/frames/"
siftdir = "/Users/ishanarya/Desktop/school/cv/ps/ps4/sift/"

print("Getting fnames.")

fnames = glob.glob(siftdir + "*.mat")
fnames = [os.path.basename(name) for name in fnames]

'''
@mat:
    im1
        positions1
        orients1
        scales1
        descriptors1
    im2
        positions2
        orients2
        scales2
        descriptors2
'''

# mat = scipy.io.loadmat("twoFrameData.mat")
# num_feats = mat["descriptors"].shape[0]

print("Getting mats.")

mats = [scipy.io.loadmat(os.path.join(siftdir, fname)) for fname in fnames]
# mats = np.load("mats.npy")
# mats = list(map(lambda fname: scipy.io.loadmat(
#     os.path.join(siftdir, fname)), fnames))

frames = [13, 1119, 4419, 4900]

for frame in frames:
    im = imageio.imread(os.path.join(framesdir, fnames[frame][:-4]))
    print(
        "now use the mouse to draw a polygon, right click or double click to end it",
        flush=True,
    )
    plt.imshow(im)
    roi = roipoly(color="r")
    indices = roi.get_indices(im, mats[frame]["positions"])
    region = np.stack((roi.all_x_points, roi.all_y_points), axis=-1)

    region_name = "region" + str(frame) + ".npy"
    points_name = "points" + str(frame) + ".npy"

    np.save(region_name, region)
    np.save(points_name, indices)


# im1 = mat["im1"]

# print(mats[0])

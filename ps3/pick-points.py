import os
import imageio
import argparse
import numpy as np

#############################################################################
# TODO: Add additional imports
#############################################################################

import scipy
import matplotlib.pyplot as plt

#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################


def get_parser():
    parser = argparse.ArgumentParser(description="Points Selection")
    parser.add_argument("image1", type=str, help="path to image 1")
    parser.add_argument("image2", type=str, help="path to image 2")
    return parser


totalClicks = {}
global figure


def onclick(event):

    x = event.xdata
    y = event.ydata
    ax = event.inaxes

    if x == None:
        return

    totalClicks[ax].append((x, y))

    ax.plot(x, y, "x", mew=2, ms=5)
    figure.canvas.draw()


def pick_points(img1, img2):
    """
    Functionality to get manually identified corresponding points from two views.

    Inputs:
    - img1: The first image to select points from
    - img2: The second image to select points from

    Output:
    - coords1: An ndarray of shape (N, 2) with points from image 1
    - coords2: An ndarray of shape (N, 2) with points from image 2
    """
    ############################################################################
    # TODO: Implement pick_points
    ############################################################################
    global figure
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img1)
    ax[1].imshow(img2)

    totalClicks[ax[0]] = []
    totalClicks[ax[1]] = []

    figure = f

    f.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    coords1 = np.asarray(totalClicks[ax[0]])
    coords2 = np.asarray(totalClicks[ax[1]])

    print("Coords1:", coords1)
    print("Coords2:", coords2)

    return coords1, coords2

    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################


if __name__ == "__main__":
    args = get_parser().parse_args()

    img1 = np.asarray(imageio.imread(args.image1))
    img2 = np.asarray(imageio.imread(args.image2))

    coords1, coords2 = pick_points(img1, img2)

    assert len(coords1) == len(
        coords2), "The number of coordinates does not match"

    filename1 = os.path.splitext(args.image1)[0] + ".npy"
    filename2 = os.path.splitext(args.image2)[0] + ".npy"

    assert not os.path.exists(
        filename1), f"Output file {filename1} already exists"
    assert not os.path.exists(
        filename2), f"Output file {filename2} already exists"

    np.save(filename1, coords1)
    np.save(filename2, coords2)

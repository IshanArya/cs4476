import numpy as np


def dist2(x, c):
    ndata, dimx = x.shape
    ncentres, dimc = c.shape
    if dimx != dimc:
        raise NameError("Data dimension does not match dimension of centres")

    n2 = (
        np.transpose(
            np.dot(
                np.ones((ncentres, 1)),
                np.transpose(np.sum(np.square(x), 1).reshape(ndata, 1)),
            )
        )
        + np.dot(
            np.ones((ndata, 1)),
            np.transpose(np.sum(np.square(c), 1).reshape(ncentres, 1)),
        )
        - 2 * np.dot(x, np.transpose(c))
    )

    n2[n2 < 0] = 0
    return n2


def match_descriptors(desc1, desc2):
    """ Finds the `descriptors2` that best match `descriptors1`
    
    Inputs:
    - desc1: NxD matrix of feature descriptors
    - desc2: MxD matrix of feature descriptors

    Returns:
    - indices: the index of N descriptors from `desc2` that 
               best match each descriptor in `desc1`
    """
    N = desc1.shape[0]
    indices = np.zeros((N,), dtype="int64")

    ############################
    # TODO: Add your code here #
    ############################

    ssd = dist2(desc1, desc2)

    indices = np.argmin(ssd, axis=1)

    ############################
    #     END OF YOUR CODE     #
    ############################

    return indices


def calculate_bag_of_words_histogram(vocabulary, descriptors):
    """ Calculate the bag-of-words histogram for the given frame descriptors.
    
    Inputs:
    - vocabulary: kxd array representing a visual vocabulary
    - descriptors: nxd array of frame descriptors
    
    Outputs:
    - histogram: k-dimensional bag-of-words histogram
    """
    k = vocabulary.shape[0]
    histogram = np.zeros((k,), dtype="int64")

    ############################
    # TODO: Add your code here #
    ############################

    ssd = dist2(descriptors, vocabulary)

    nearest_vocabs = np.argmin(ssd, axis=1)

    histogram = np.bincount(nearest_vocabs)

    histogram = np.append(histogram, np.zeros(
        (k - histogram.size), dtype=np.int_))

    ############################
    #     END OF YOUR CODE     #
    ############################

    return histogram


def caculate_normalized_scalar_product(hist1, hist2):
    """ Caculate the normalized scalar product between two histograms.
    
    Inputs:
    - hist1: k-dimensional array
    - hist2: k-dimensional array
    
    Outputs:
    - score: the normalized scalar product described above
    """

    score = 0

    ############################
    # TODO: Add your code here #
    ############################

    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)

    norm = norm1 * norm2

    dot = np.dot(hist1, hist2).astype(np.float_)

    score = dot/norm

    ############################
    #     END OF YOUR CODE     #
    ############################

    return score

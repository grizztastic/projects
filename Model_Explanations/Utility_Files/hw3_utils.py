import inspect
import warnings
import os
from collections import namedtuple

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from vgg16 import vgg16
from imagenet_classes import class_names
import scipy

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.backend import resize_images

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


Domains = namedtuple("Domains", ["X", "Y", "Pred", "Idx"])


def get_model(sess):
    """Return the pretained model

    Arguments:
        sess {tf.Session} -- Tensorflow session

    Returns:
        vgg16 -- A class of VGG16 model

    Example Usage:
    >>> sess = tf.Session()
    >>> model = get_model(sess)
    """
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    return vgg16(
        imgs, 'imagenet', sess
    )
    # Download from https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz


def load_ImageNet():
    for d in ['.', '..', '../..', '/autograder/source/tests']:
        try:
            path = os.path.join(d, "imagenet_14c_vgg16.npz")
            assert (os.path.exists(path) and os.access(path, os.R_OK))
            break
        except Exception:
            continue

    all = np.load(path)

    X = all['X']
    Y = all['Y']
    Pred = all['Pred']
    Idx = all['Idx']

    return Domains(X, Y, Pred, Idx)


def get_tensor_names():
    """Return the names of all tensors in the graph
    """
    return [n.name + ":0" for n in tf.get_default_graph().as_graph_def().node]


def ImageNet_name2idx(name):
    """Given a name, return the class index.
    """
    return class_names.index(name)


def ImageNet_idx2name(idx):
    """Given an index, return the corresponding name.
    """
    return class_names[idx]


def binary_mask(X,
                attr,
                channel_first=False,
                norm=True,
                threshold=0.2,
                blur=0,
                background=0.1):
    """Visualize the attribution map by overlapping it with the input

    Arguments:

        X {np.ndarray} -- Input dataset. A shape of (N x H x W x C) is required

        attr {[type]} -- attribution map A shape of (N x H x W x C) is required

    Keyword Arguments:

        channel_first {bool} -- If True, the channle dimension is the first
        dimension (default: {False})

        norm {bool} -- If True, normalize the attribution score to [0, 1]
        (default: {True})

        threshold {float} -- Thresholding the attribution scores (default:
        {0.2})

        blur {int} -- The Gaussion blur coefficient. If set to 0, no blur is
        applied (default: {0})

        background {float} -- The brightness of the pixels with zero
        attribution scores (default: {0.1})

    Returns:
        np.ndarray -- Final visualization (N x W x H x C)

    """
    if len(attr.shape) == 4:
        attr = np.mean(attr, axis=-1, keepdims=True)
    else:
        raise ValueError("The input attribution map must contain 4 dimensions")

    if len(X.shape) != 4:
        raise ValueError("The input must contain 4 dimensions")

    if channel_first:
        X = np.transpose(X, axis=(0, 2, 3, 1))

    result = []
    for score, image in zip(attr, X):
        if norm:
            score = score / (np.max(score) + 1e-9)
        if blur > 0:
            score = scipy.ndimage.filters.gaussian_filter(score, blur)
            score = score / (np.max(score) + 1e-9)

        score[score >= threshold] = 1.0
        score[score <= threshold] = 0

        score[score == 0.0] = background

        binary_map = image * score
        binary_map = 255 * binary_map / (np.max(binary_map) + 1e-9)
        binary_map = np.uint8(binary_map)
        result.append(binary_map[None, :])
    result = np.vstack(result)
    return result


def point_cloud(grads, threshold=None):
    if len(grads.shape) == 4:
        grads = np.mean(grads, axis=-1)

    if threshold is not None:
        grads[grads < threshold] = 0

    grads /= (np.max(grads, axis=(1, 2), keepdims=True) + 1e-9)
    return grads


def exercise(andrew_username=None, seed=None):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    cname = inspect.getouterframes(inspect.currentframe(), 2)[1][3]

    seedh = hash(hash(andrew_username) + hash(cname) + hash(seed)) % 0xffffffff

    np.random.seed(seedh)
    tf.compat.v1.set_random_seed(seedh)

    print(f"=== EXERCISE {cname} {andrew_username} {seed:x} {seedh:x} ===")

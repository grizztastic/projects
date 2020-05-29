import tensorflow as tf
import numpy as np
from hw3_utils import binary_mask, point_cloud

# Feel free to add more helper functions


def AverageDrop(sess, attr_map, input,
                input_tensor, output_tensor,
                target_class):
    """Implementation of Average Drop %

    https://arxiv.org/abs/1710.11063

    Arguments:
        sess {tf.Session} -- tensorflow session
        attr_map {np.ndarray} -- NxHxW, the attribution maps for input
        input {np.ndarray} -- NxHxWxC, the input dataset
        input_tensor {tf.Tensor} -- Symbolic tensor of input node
        output_tensor {tf.Tensor} -- Symbolic tensor of pre-softmax output
        target_class: {int} -- Class of Interest

    Returns:
        float -- average drop % score
    """
    score = None
    # >>> Your code starts here <<<
    N = input.shape[0]
    attr_map = np.expand_dims(attr_map, axis=attr_map.ndim) #expand dimensions at end to fit into binary mask
    Ma_x = binary_mask(input, attr_map, False, True, 0, 0.0, 0.0) #get masked values
    attr_score = sess.run(output_tensor[:, target_class], feed_dict={input_tensor: input}) #output of input tensor at pre-softmax layer
    mask_score = sess.run(output_tensor[:, target_class], feed_dict={input_tensor: Ma_x})  #output of masked input at pre-softmax layer
    idx_subtract = attr_score - mask_score #input - mask for comparison
    score1 = np.maximum(idx_subtract, 0) / attr_score #score equation still needing to be summed
    score = 100 * np.sum(score1) / N  #divide by num images
    # >>> Your code ends here <<<
    return score



def N_Ord(sess, attr_map, input,
          input_tensor, output_tensor,
          target_class):
    """Implementation of Necessity Ordering

    https://arxiv.org/abs/2002.07985

    Arguments:
        sess {tf.Session} -- tensorflow session
        attr_map {np.ndarray} -- HxW, the attribution map for input
        input {np.ndarray} -- HxWxC, the input image
        input_tensor {tf.Tensor} -- Symbolic tensor of input node
        output_tensor {tf.Tensor} -- Symbolic tensor of pre-softmax output
        target_class: {int} -- Class of Interest

    Returns:
        float -- Necessity Drop score
    """
    score = None
    # >>> Your code starts here <<<

    # >>> Your code ends here <<<
    return score

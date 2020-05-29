import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense
from tensorflow import reduce_mean
import hw2_utils


def build_model(X, out_dim, is_training=True):
    """
    Build a specific model here

    Arguments:
        X {tf.keras.Input} -- The symbolic input tensor
        out_dim {int} -- The number of classes

    Keyword Arguments:
        is_training {bool} -- If True, set some layers to training mode
        (default: {True})

    Returns:
        tf.Tensor, tf.Tensor -- logit: pre-softmax scores. probit: post-softmax
        scores
    """

    hw2_utils.exercise(
        andrew_username="agrizzaf", # <<< set your andrew username here
        seed=42
    )

    logit = None
    probit = None
    '''Implementation of model described in section 7.1'''
    input = X
    conv1 = Conv2D(filters=32, kernel_size=(7, 7), padding='valid', activation='relu')(input)
    batch_1 = BatchNormalization(trainable=is_training)(conv1)
    maxpool_1 = MaxPool2D(pool_size=(2,2),strides=2)(batch_1)
    flatten = Flatten()(maxpool_1)
    dense_1 = Dense(units=1024, activation='relu')(flatten)
    logit = Dense(units=out_dim)(dense_1)
    probit = tf.nn.softmax(logit)

    # >>> End of your code <<<

    return logit, probit


def build_loss_optimizer(Y, logit):
    """
    Compute loss and build optimizer using tf.train

    Arguments:
        Y {tf.Tensor} -- The symbolic tensor of labels
        logit {tf.Tensor} -- Pre-softmax scores

    Returns:
        tf.Tensor, tf.Optimizer -- loss and optimzer
    """

    hw2_utils.exercise(
        andrew_username="agrizzaf", # <<< set your andrew username here
        seed=42
    )

    mean_loss = None
    optimizer = None

    '''Code used from homework part 1'''
    # >>> Start of your code <<<
    mean_loss = reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)

    # >>> End of your code <<<

    return mean_loss, optimizer

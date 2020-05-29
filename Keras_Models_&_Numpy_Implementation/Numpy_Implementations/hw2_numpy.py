# %%
# Part II Numpy implementations
import numpy as np
import hw2_utils
import sys
np.set_printoptions(threshold=sys.maxsize)

'''Function used to pad the input x with "pad" extra zeros around the edges. Code was derived from my Introduction to DL 
hw0 and the np.pad documentation online. '''
def zero_pad(x, pad):
    return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),'constant', constant_values=(0))

def convolution(x, w, b, stride, pad):
    """
    4.1 Understanding Convolution
    Forward Pass for a convolution layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

    Output:
    - out: Output of the forward pass
    """

    hw2_utils.exercise(
        andrew_username="agrizzaf", # <<< set your andrew username here
        seed=42
    )

    out = None
    ###########################################################################
    # Your code starts here
    ###########################################################################
    # TODO 4.1 Understanding Convolution
    #print(x)
    Xpad = zero_pad(x, pad)  #zero pad the entire input
    batch_size, num_channels, image_height, image_width = Xpad.shape #acquire the necessary values to solve forward pass from shape of padded input
    num_filters, filter_height, filter_width = w.shape[0], w.shape[2], w.shape[3]

    height_output = (image_height - filter_height + stride) // stride  # calculation for height output size
    width_output = (image_width - filter_width + stride) // stride  # calculation for width output size
    out = np.zeros((batch_size, num_filters, height_output, width_output))  # initialize output array for forward pass through conv layer
    '''For loop to iterate over all batches and filters while scanning the kernel and summing up convolution over all
    input channels to give forward pass output. This code was derived from the psuedo-code from Introduction to Deep Learning,
    a previous homework that I computed a forward pass of a 1-D convolution from IDL, and the cs231 Stanford website giving
    visuals of how a convolution works.'''
    for batch in range(batch_size):         #Iterate over the batches
        xpad_prev = Xpad[batch]             #get the data over one batch
        for filter in range(num_filters):   #iterate over the number of filters
            h = 0                           #used to keep track of the output value height index for filter
            for height in range(0, height_output, stride): #iterate over height index for input channels to be summed
                j = 0                       #used to keep track of the output value width index for filter
                for width in range(0, width_output, stride): #iterate over width index for input channels to be summed
                    #calculate the convolution value over the input channels and place in the proper output of the output matrix (W*x)
                    out[batch][filter][h][j] += \
                        np.sum(np.multiply(xpad_prev[:, height:height + filter_height, width:width + filter_width], w[filter, :, :, :]))
                    j += 1
                h += 1
            out[batch][filter][:][:] += b[filter] #add the bias to the output value
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return out

def grayscale_filter(w):
    """ your grayscale filter

    Modify the second filter of the input filters as a grayscale filter

    Input:
    - w: Conv filter of shape [2, 3, 3, 3]
    - w: Filter weights of shape (F, C, HH, WW)

    Output:
    - w: The modified filter

    """

    hw2_utils.exercise(
        andrew_username="agrizzaf", # <<< set your andrew username here
        seed=42
    )

    ###########################################################################
    # Your code starts here
    ###########################################################################
    '''Update second filter with Luma encoding. Only set middle value as to analyze the specified input element 
    through each channel and acquire proper linear equation output'''
    w[1, 0, 1, 1] = 0.299
    w[1, 1, 1, 1] = 0.587
    w[1, 2, 1, 1] = 0.114
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################
    return w


def relu(x):
    """
    4.2 ReLU Implementation
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    """

    hw2_utils.exercise(
        andrew_username="agrizzaf", # <<< set your andrew username here
        seed=42
    )

    out = None

    ###########################################################################
    # Your code starts here
    ###########################################################################
    # TODO: 4.2 ReLU Implementation

    out = np.maximum(0, x)  #Relu is the max of 0 or input for every element in input matrix
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return out


def max_pool(x, pool_param):
    """
    4.3 MaxPooling Implementation
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    """

    hw2_utils.exercise(
        andrew_username="agrizzaf", # <<< set your andrew username here
        seed=42
    )

    out = None

    ###########################################################################
    # Your code starts here
    ###########################################################################
    # TODO: 4.3 MaxPooling Implementation
    '''Get required values needed to perform maxpool'''
    batch_size, num_channels, image_height, image_width = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    height_output = (image_height - pool_height + stride) // stride  # calculation for height output size
    width_output = (image_width - pool_width + stride) // stride  # calculation for height output size
    out = np.zeros((batch_size, num_channels, height_output, width_output))  # initialize out for forward pass through maxpool layer
    for batch in range(batch_size):       # iterate over the number of batches
        for chan in range(num_channels):  # iterate over the number of input channels
            for height in range(0, height_output, stride):
                for width in range(0, width_output, stride):
                    '''Maxpool takes the maximum value of the input the kernel is over and places it in the output matrix'''
                    x_slice = x[batch, chan, height:height + pool_height, width:width + pool_width]
                    out[batch][chan][height][width] = np.max(x_slice)
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return out


def dropout(x, mode, p):
    """
    4.4  Dropout Implementation
    Performs the forward pass for (inverted) dropout.
    Inputs:
    - x: Input data, of any shape
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
    if the mode is test, then just return the input.
    Outputs:
    - out: Array of the same shape as x.
    """

    hw2_utils.exercise(
        andrew_username="agrizzaf", # <<< set your andrew username here
        seed=42
    )

    out = None

    if mode == 'train':
        #######################################################################
        # Your code starts here
        #######################################################################
        # TODO: 4.4 Train mode of dropout
        '''Dropout rate uses binomial distribution to determine whether a node is turned on or off (1,0)'''
        dropout_vals = np.random.binomial(1, (1-p), size=x.shape)/(1-p)
        out = x * dropout_vals #multiply input by the dropout mask to enact whether the input is on or off
        #######################################################################
        # END OF YOUR CODE
        #######################################################################

    elif mode == 'test':
        #######################################################################
        # Your code starts here
        #######################################################################
        # TODO: 4.4 Test mode of dropout

        out = x

        #######################################################################
        # END OF YOUR CODE
        #######################################################################

    return out

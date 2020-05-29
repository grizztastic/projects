import tensorflow as tf
import numpy as np

class SaliencyMap(object):
    """Implementaion of Saliency Map with Vanilla Gradient

        https://arxiv.org/pdf/1312.6034.pdf

        Example Usage:
        ## >>> attr_fn = SaliencyMap(sess, input_tensor, output_tensor, 0)
        ## >>> saliency_map = attr_fn(X) <--- equivalent to attr_fn.__call__(X)
    """
    def __init__(self, sess, input_tensor, output_tensor, target_class):
        """__Constructor__

        Arguments:
            sess {tf.Session} -- Tensorflow session
            input_tensor {tf.Tensor} -- Symbolic tensor of input node
            output_tensor {tf.Tensor} -- Symbolic tensor of pre-softmax output
            target_class {int} -- The class of interest to run explanation.
        """
        self.sess = sess
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.target_class = target_class

        self._define_ops()

    def _define_ops(self):
        """Add whatever operations you feel necessary into the computational
        graph.
        """

        # >>> Your code starts here <<<
        y_s = self.output_tensor[:, self.target_class]
        x_s = self.input_tensor
        self.gradients = tf.gradients(ys=y_s, xs=x_s)[0] #generate symbolic tensor for gradient calculation
        self.multiplied_gradients = tf.multiply(self.input_tensor, self.gradients) #generate symbolic tensor for multiplied gradient calculation
        # >>> Your code ends here <<<

    def __call__(self, X, batch_size=16, multiply_with_input=True):
        """__call__ forward computation to generate the saliency map

        Arguments:
            X {np.ndarray} -- Input dataset

        Keyword Arguments:
            batch_size {int} -- Batch size (default: {16})
            multiply_with_input {bool} -- If True, return grad x input,
                else return grad (default: {True})

        Returns np.ndarray of the same shape as X.
        """
        # >>> Your code starts here <<<
        gradients, multiplied_gradients = self.sess.run([self.gradients, self.multiplied_gradients], feed_dict={self.input_tensor: X}) #calculate gradients and multiplied gradients
        if multiply_with_input:
            return multiplied_gradients
        else:
            return gradients

class IntegratedGrad(object):
    """Implementaion of Integrated Gradient
        https://arxiv.org/pdf/1703.01365.pdf
        Example Usage:
       # >>> attr_fn = IntegratedGrad(sess, input_tensor, output_tensor, 0)
       # >>> integrated_grad = attr_fn(X, 'black')
    """

    def __init__(self, sess, input_tensor, output_tensor, target_class):
        """_Constructor_
        Arguments:
            sess {tf.Session} -- Tensorflow session
            input_tensor {tf.Tensor} -- Symbolic tensor of input node
            output_tensor {tf.Tensor} -- Symbolic tensor of pre-softmax output
            target_class {int} -- The class of interest to run explanation.
        """
        np.random.seed(1111)
        self.sess = sess
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.target_class = target_class
        self._define_ops()

    def _define_ops(self):
        """Add whatever operations you feel necessary into the computational
        graph.
        """
        # >>> Your code starts here <<<
        self.x_baseline = tf.placeholder(tf.float32, shape=self.input_tensor.get_shape()[1:]) #symbolic tensor for x baseline
        self.alpha = tf.placeholder(tf.float32, shape=()) #symbolic tensor for alpha
        self.step = self.x_baseline + (self.input_tensor - self.x_baseline) * self.alpha  #symbolic tensor for evaluating a step calculation for the integral/summation
        self.gradient = tf.gradients(self.output_tensor[:, self.target_class], self.input_tensor)[0] #symbolic tensor for calculating the gradient
        self.multiplied_gradient = tf.multiply(self.gradient, self.input_tensor - self.x_baseline) #symbolic tensor for calculating multiplied gradient
        # >>> Your code ends here <<<

    def __call__(self,
                 X,
                 baseline='black',
                 batch_size=16,
                 num_steps=50,
                 multiply_with_input=True):
        """_call_ forward computation to generate the integrated gradient
        Arguments:
            X {np.array} -- Input dataset
        Keyword Arguments:
            baseline {str} -- The baseline input. One of 'black', 'white',
                or 'random' (default: {'black'})
            batch_size {int} -- Batch size (default: {16})
            num_steps {int} -- resolution of using sum to approximate the
                integral (default: {50})
            multiply_with_input {bool} -- If True, return grad x input,
                else return grad (default: {True})
        Returns np.ndarray of the same shape as X.
        """
        # >>> Your code starts here <<<
        '''Code for generating x_baseline based on the input argument baseline'''
        if baseline == 'black':
            x_baseline = np.zeros(shape=(self.input_tensor.get_shape()[1:]))
        elif baseline == 'white':
            x_baseline = np.ones(shape=(self.input_tensor.get_shape()[1:]))
        elif baseline == 'random':
            x_baseline = np.random.normal(0, 0.1, self.input_tensor.get_shape()[1:])

        total_gradient = None
        for i in range(num_steps):
            alpha = (i + 1) / num_steps #alpha value (i/N in paper)
            x_step = self.sess.run(self.step, {self.x_baseline: x_baseline, self.alpha: alpha, self.input_tensor: X}) #generate single step value
            gradient = self.sess.run(self.gradient, {self.input_tensor: x_step}) #caluculate the gradient at that step value
            if total_gradient is None:
                total_gradient = gradient
            else:
                total_gradient += gradient #sum up the gradient at the step value

        if multiply_with_input:
            return (X - x_baseline) * total_gradient / num_steps #multiply with input minus baseline as paper says
        else:
            return total_gradient / num_steps #no multiply with input

import numpy as np
import tensorflow as tf

from typing import Tuple

from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import \
    Conv2D, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Input, Model

from hw4_mnist import MNISTModel


class MNISTModelRegular(MNISTModel):
    """A version of an MNIST model to instrument the pre-activation output of the
       last intermediate layer (the one before the one that produces logits,
       the softmax layer has no trainable parameters) and adds L2
       regularization to the learnable parameters of this layer. We refer to the
       output of this layer as "features".
    """

    def __init__(self, lam=0.1, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

        self.lam = lam

    def build(self):
        # Running this will reset the model's parameters
        layers = []

        # >>> Your code here <<<
        '''Code below taken from the hw4_mnist file to generate layers'''
        layers.append(Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), padding="same", activation='relu'))
        layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        layers.append(Conv2D(filters=32, kernel_size=(4, 4),padding='same', activation='relu'))
        layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        layers.append(Flatten())
        layers.append(Dense(64, activation='relu'))
        layers.append(Dense(self.num_classes, name="features", kernel_regularizer=tf.keras.regularizers.l2(self.lam)))  # pre-softmax layer with l2 kernel regularizer
        layers.append(Activation('softmax'))  # post-softmax outputs form the l2 feature loss layer
        # >>> End of your code <<<

        self.layers = layers

        self._define_ops()
    '''Define ops function taken from mnist model file'''
    def _define_ops(self):
        self.X = Input(shape=(28, 28, 1))
        self.Ytrue = Input(shape=(self.num_classes), dtype=tf.int32)

        self.tensors = self.forward(self.X, self.Ytrue)

        self.model = Model(self.X, self.tensors['probits'])

        # models can be symbolically composed
        self.logits = Model(self.X, self.tensors['features'])
        self.features = Model(self.X, self.tensors['features'])
        self.probits = Model(self.X, self.tensors['probits'])
        self.preds = Model(self.X, self.tensors['preds'])

        # functions evaluate to concrete outputs
        self.f_logits = K.function(self.X, self.tensors['features'])
        self.f_features = K.function(self.X, self.tensors['features'])
        self.f_probits = K.function(self.X, self.tensors['probits'])
        self.f_preds = K.function(self.X, self.tensors['preds'])

    '''Forward function generated as per the mnist model file with loss and features added'''
    def forward(self, X, Ytrue=None):
        _features: tf.Tensor = None # new tensor to build
        _logits: tf.Tensor = None
        _probits: tf.Tensor = None
        _preds: tf.Tensor = None
        _loss: tf.Tensor = None

        # >>> Your code here <<<
        c = X
        parts = []
        for l in self.layers:
            c = l(c)
            parts.append(c)

        _logits = parts[-2] #logits layer
        _features = parts[-2] #features layer
        _probits = parts[-1] #probits layer

        _preds = tf.argmax(_probits, axis=1) #predictions of probits (uses feature layer)

        if Ytrue is not None:
            _loss = K.mean(K.sparse_categorical_crossentropy(
                self.Ytrue,
                _probits
            ))
        # >>> End of your code here <<<

        return {
            'features': _features,
            'logits': _logits,
            'probits': _probits,
            'preds': _preds,
            'loss': _loss,
        }

    def load(self, batch_size=16, filename=None):
        if filename is None:
            filename = f"model.regular{self.lam}.MNIST.h5"

        super().load(batch_size=batch_size, filename=filename)

    def save(self, filename=None):
        if filename is None:
            filename = f"model.regular{self.lam}.MNIST.h5"

        super().save(filename=filename)


class Representer(object):
    def __init__(
        self,
        model: MNISTModelRegular,
        X: np.ndarray,
        Ytrue: np.ndarray
    ) -> None:
        """
        X: np.ndarray [N, 28, 28, 1] training points
        Y: np.ndarray [N] ground truth labels
        """

        assert "features" in model.tensors, \
            "Model needs to provide features tensor."
        assert "loss" in model.tensors, \
            "Model needs to provide loss tensor."
        assert "logits" in model.tensors, \
            "Model needs to provide logits tensor."

        self.model = model
        self.lam = model.lam
        self.X = X
        self.Ytrue = Ytrue

        self._define_ops()

    def _define_ops(self):

        # >>> Your code here <<<
        self.features_probits_pred_X = self.model.probits(self.X) #generate the post softmax predictions using feature layer
        self.features_pred_X = self.model.features(self.X) #generate the features
        # >>> End of your code <<<

    def similarity(self, Xexplain: np.ndarray) -> np.ndarray:
        """For each input instance, compute the similarity between it and every one of
        the training instances. This is the f_i f_t^T term in the paper.

        inputs:
            Xexplain: np.ndarray [M, 28, 28, 1] -- images to explain

        return
            np.ndarray [M, N]

        """

        # >>> Your code here <<<
        features_pred_Xeplain = self.model.features(Xexplain) #get features of Xexplain
        f_i = self.model.session.run(features_pred_Xeplain) #run session to generate the features for Xexplain
        f_tT = self.model.session.run(self.features_pred_X).T #run session to generate the features for X, then transposed
        return np.dot(f_i, f_tT)
        # >>> End of your code <<<

    def coeffs(self) -> np.ndarray:
        """For each training instance, compute its representer value coefficient. This
        is the alpha term in the paper.

        inputs:
            none

        return
            np.ndarray [N, 10]

        """

        # >>> Your code here <<<
        one_hot_Y = np.eye(self.model.num_classes)[self.Ytrue] #ground truth array
        dL = self.model.session.run(self.features_probits_pred_X) - one_hot_Y #derivative of loss w.r.t. probits
        numerator = dL
        denominator = (-2.0 * self.lam * len(self.Ytrue))
        return numerator / denominator

        # >>> End of your code <<<

    def values(self, coeffs, sims):
        """Given the training instance coefficients and train/test feature
        similarities, compute the representer point values. This is the k term
        from the paper.

        inputs:
            coeffs: np.ndarray [N, 10]
            sims: np.ndarray [M, N]

        return
            np.ndarray [M, N, 10]

        """

        # >>> Your code here <<<
        return np.einsum('ij,ki->kij', coeffs, sims) #use einsum to generate the proper array dimensions
        # >>> End of your code <<<

    def coeffs_and_values(
        self,
        Xexplain: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """For each input instance, compute the representer point coefficients of the
        training data(self.X, self.Y)

        inputs:
            Xexplain: np.ndarray [M, 28, 28, 1] -- images to explain
            target: target class being explained

        returns:
            coefficients: np.ndarray of size [N, 10]
            values: np.ndarray of size [M, N, 10]
              N is size of |self.X|

        """

        coeffs = self.coeffs()
        sims = self.similarity(Xexplain)

        return coeffs, self.values(coeffs, sims)
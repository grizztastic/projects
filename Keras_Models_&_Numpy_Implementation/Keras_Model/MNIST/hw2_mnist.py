import numpy as np
from tensorflow import keras
from tensorflow.keras import Input, layers, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import hw2_utils
from hw2_utils import Splits

# Load data and reshape to 28x28x1 (1 channel). Training, validation, test data
# is then splits.train, splits.valid, splits.test . Within a split, inputs are
# split.X and outputs are split.Y.
splits = Splits(*[
    split._replace(
        X=np.reshape(split.X, (split.X.shape[0], 28, 28, 1))
    )
    for split in hw2_utils.load_mnist()])


def build_model_mnist1():
    """
    Create model as described in exercise. Return the tensorflow.keras.model
    object.
    """

    hw2_utils.exercise(
        andrew_username="agrizzaf", # <<< set your andrew username here
        seed=42
    )

    input = Input(shape=(28, 28, 1))

    model = None

    # >>> Start of your code <<<
    '''Implementation of model described in section 5.1'''
    conv_1 = Conv2D(32, (5, 5), 1, activation= 'relu')(input)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_1)
    dropout_1 = Dropout(rate=0.2)(maxpool_1)
    flatten = Flatten()(dropout_1)
    dense = Dense(128, activation='relu')(flatten)
    output = Dense(10,activation='softmax')(dense)
    model = Model(inputs=input, outputs=output)
    # >>> End of your code <<<
    return model


def train_model_mnist1(model, X, Y):
    """ Train the given model over the given instances. """

    hw2_utils.exercise(
        andrew_username="agrizzaf", # <<< set your andrew username here
        seed=42
    )



    # >>> Start of your code <<<

    # Please use verbosity=2 for model.fit when generating evaluation log for
    # submission. DO NOT artificially inflate the training time for your first
    # model in order to pass the final test case.
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(X, keras.utils.to_categorical(Y), batch_size=64, verbose=2, epochs=3, validation_split=0.2)
    # >>> End of your code <<<

    return model


def build_model_mnist2():
    """Create model as described in exercise. Return the tensorflow.keras.model
    object."""

    hw2_utils.exercise(
        andrew_username="agrizzaf", # <<< set your andrew username here
        seed=42
    )

    input = Input(shape=(28, 28, 1))

    model = None

    # >>> Start of your code <<<
    '''My implementation of a model that achieves better accuracy and converges faster than the model required above.'''
    conv_1 = Conv2D(32, (5, 5), 1, padding='same', activation='relu')(input)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_1)
    dropout_1 = Dropout(rate=0.4)(maxpool_1)
    flatten = Flatten()(dropout_1)
    dense1 = Dense(128, activation='relu')(flatten)
    dropout_2 = Dropout(0.2)(dense1)
    output = Dense(10, activation='softmax')(dropout_2)
    model = Model(inputs=input, outputs=output)
    # >>> End of your code <<<

    return model


def train_model_mnist2(model, X, Y):
    """ Train the given model over the given instances. """

    hw2_utils.exercise(
        andrew_username="agrizzaf", # <<< set your andrew username here
        seed=42
    )

    # >>> Start of your code <<<

    # Please use verbosity=2 for model.fit when generating evaluation log for
    # submission.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, keras.utils.to_categorical(Y), batch_size=128, verbose=2, epochs=2, validation_split=0.2)
    # >>> End of your code <<<

    return model

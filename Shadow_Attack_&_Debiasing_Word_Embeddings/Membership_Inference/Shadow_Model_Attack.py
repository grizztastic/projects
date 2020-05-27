import hw5_part1_utils

from typing import Tuple
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np

from tqdm import tqdm


# YOUR IMPLEMENTATION FOR THE SHADOW MODEL ATTACK GOES HERE ###################


def synthesize_attack_data(
        target_model: hw5_part1_utils.TargetModel,
        shadow_data: np.ndarray,
        shadow_labels: np.ndarray,
        num_shadow_models: int = 4
):
    """Synthesize attack data.

    Arguments:

        target_model {TargetModel} -- an instance of the TargetModel class;
          behaves as a keras model but additionally has a train_shadow_model
          function, which takes a subset of the shadow data and labels and
          returns a model with identical architecture and hyperparameters to
          the original target model, but that is trained on the given shadow
          data.

        shadow_data {np.ndarray} -- data available to the attack to train
          shadow models. If the arget model's training set is size N x D,
          shadow_data is 2N x D.

        shadow_labels {np.ndarray} -- the corresponding labels to the
          shadow_data, given as a numpy array of 2N integers in the range 0 to
          C where C is the number of classes.

        num_shadow_models {int} -- the number of shadow models to use when
          constructing the attack model's dataset.

    Returns: three np.ndarrays; let M = 2N * num_shadow_models

        attack_data {np.ndarray} [M, 2C] -- shadow data label probability and
           label one-hot

        attack_classes {np.ndarray} [M, 1 of {0,1,...,C-1}] -- shadow data
           labels

        attack_labels {np.ndarray} [M, 1 of {0,1}] -- attack data labels
           (training membership)

    """

    C = shadow_labels.max() + 1

    attack_data: np.ndarray = None
    attack_classes: np.ndarray = None
    attack_labels: np.ndarray = None

    # SOLUTION
    A = []
    S_classes = []
    S_labels = []
    for i in range(num_shadow_models):
        data_split = hw5_part1_utils.DataSplit(shadow_labels, np.random.seed(i))  # get in and out indexes
        S_in_idx = data_split.in_idx
        S_out_idx = data_split.out_idx

        S_in = shadow_data[S_in_idx]  # get the S in data using the S in indexes for the shadow data
        S_in_labels = shadow_labels[S_in_idx]  # get the S in labels using the S in indexes for the shadow labels
        S_in_labels_one_hot = to_categorical(S_in_labels, 10)  # get one hot encondings of S in labels

        S_out = shadow_data[S_out_idx]  # get the S out data using the S out indexes for the shadow data
        S_out_labels = shadow_labels[S_out_idx]  # get the S out labels using the S out indexes for the shadow labels
        S_out_labels_one_hot = to_categorical(shadow_labels[S_out_idx], 10)  # get one hot encondings of S out labels

        trained_model = target_model.train_shadow_model(S_in,
                                                        S_in_labels)  # train the shadow model with generated S in and S in labels data
        S_in_preds = trained_model.predict(S_in)  # predict the output from trained model using S in data
        S_out_preds = trained_model.predict(S_out)  # predict the output from trained model using S out data
        A_in = np.hstack((S_in_preds, S_in_labels_one_hot))  # get A in data by stacking the preds and labels
        A_out = np.hstack((S_out_preds, S_out_labels_one_hot))  # get A out data by stacking the preds and labels
        A.append(np.vstack((A_in, A_out)))
        S_classes.append(
            np.hstack((S_in_labels, S_out_labels)))  # append stacked S in and out labels for attack_classes
        S_labels.append(
            np.hstack((np.ones(len(S_in_labels)), np.zeros(len(S_out_labels)))))  # one hot vector for attack classes
    attack_data = np.vstack((A))
    attack_classes = np.hstack((S_classes))
    attack_labels = np.hstack((S_labels))
    # END OF SOLUTION

    return attack_data, attack_classes, attack_labels


def build_attack_models(
        target_model: hw5_part1_utils.TargetModel,
        shadow_data: np.ndarray,
        shadow_labels: np.ndarray,
        num_shadow_models: int = 4
):
    """Build attacker models.

    Arguments:

        target_model {TargetModel} -- an instance of the TargetModel class;
          behaves as a keras model but additionally has a train_shadow_model
          function, which takes a subset of the shadow data and labels and
          returns a model with identical architecture and hyperparameters to
          the original target model, but that is trained on the given shadow
          data.

        shadow_data {np.ndarray} -- data available to the attack to train
          shadow models. If the arget model's training set is size N x D,
          shadow_data is 2N x D.

        shadow_labels {np.ndarray} -- the corresponding labels to the
          shadow_data, given as a numpy array of 2N integers in the range 0 to
          C where C is the number of classes.

        num_shadow_models {int} -- the number of shadow models to use when
          constructing the attack model's dataset.

    Returns:

        {tuple} -- a tuple of C keras models, where the c^th model predicts the
        probability that an instance of class c was a training set member.

    """

    attack_data, attack_classes, attack_labels = \
        synthesize_attack_data(
            target_model,
            shadow_data,
            shadow_labels,
            num_shadow_models=4
        )

    # to return
    attack_models: Tuple[Model] = None

    C = shadow_labels.max() + 1

    # SOLUTION
    attack_models = []
    for i in range(C):  # range over class numbers
        attack_model = get_attack_architecture(C)  # initialize attack model for class
        attack_model.fit(attack_data, attack_labels)  # fit model with attack data and labels
        attack_models.append(attack_model)  # append attack model
    attack_models = tuple(attack_models)  # turn into a tuple
    # END OF SOLUTION

    return attack_models


'''Initial model architecture taken from hw5_part1_utils.py and modified to meet the 
   assigment requests'''


def get_attack_architecture(C):
    l_in = Input((2 * C,))  # input layer size based on class number
    l_inter = Dense(4 * C, activation='relu')(l_in)  # relu activation with hidden layer dependendent on class
    l_out = Dense(1, activation='sigmoid')(l_inter)  # output layer of size 1

    m = Model(l_in, l_out)  # initialize model

    m.compile(
        loss='binary_crossentropy',  # use binary cross entropy
        optimizer='adam',
        metrics=['accuracy'],
        experimental_run_tf_function=False
    )

    return m


def evaluate_membership(attack_models, y_pred, y):
    """Evaluate the attacker about the membership inference

    Arguments:

        attack_model {tuple} -- a tuple of C keras models, where C is the
          number of classes.

        y_pred {np.ndarray} -- an N x C numpy array with the predictions of the
          model on the N instances we are performing the inference attack on.

        y {np.ndarray} -- the true labels for each of the instances given as a
          numpy array of N integers.

    Returns:

        {np.ndarray} -- an array of N floats in the range [0,1] representing
          the estimated probability that each of the N given instances is a
          training set member.

    """

    # To return
    preds: np.ndarray = None

    # SOLUTION
    predictions = []
    attack_models = list(attack_models)
    # used to give proper y value size to model for predictions
    y_for_model = [np.hstack((yp, to_categorical(Y, 10))) for yp, Y in zip(y_pred, y)]
    for i in range(len(y)):
        if i % 100 == 0:
            print("Index: {}".format(i))
        predictions.append(attack_models[y[i]].predict(np.asmatrix(y_for_model[i])))
    preds = np.array(predictions)

    # END OF SOLUTION

    return preds


# YOU DO NOT NEED TO MODIFY THE REST OF THIS CODE. ############################


if __name__ == '__main__':
    # Load the dataset.
    data = hw5_part1_utils.CIFARData()

    # Make a target model for the dataset.
    target_model = \
        hw5_part1_utils.CIFARModel(
            epochs=48,
            batch_size=2048,
            noload=True,  # prevents loading an existing pre-trained target
            # model
        ).init(
            data.train, data.labels_train,
            # data.test, data.labels_test # validation data
        )

    tqdm.write('Building attack model...')
    attack_models = build_attack_models(
        target_model,
        data.shadow,
        data.labels_shadow
    )

    tqdm.write('Evaluating target model...')
    y_pred_in = target_model.predict(data.train)
    y_pred_out = target_model.predict(data.test)

    tqdm.write('  Train Accuracy: {:.4f}'.format(
        (y_pred_in.argmax(axis=1) == data.labels_train).mean()))
    tqdm.write('  Test Accuracy:  {:.4f}'.format(
        (y_pred_out.argmax(axis=1) == data.labels_test).mean()))

    in_preds = evaluate_membership(
        attack_models,
        y_pred_in,
        data.labels_train
    )
    out_preds = evaluate_membership(
        attack_models,
        y_pred_out,
        data.labels_test
    )

    wrongs_in = y_pred_in.argmax(axis=1) != data.labels_train
    wrongs_out = y_pred_out.argmax(axis=1) != data.labels_test

    true_positives = (in_preds > 0.5).mean()
    true_negatives = (out_preds < 0.5).mean()
    attack_acc = (true_positives + true_negatives) / 2.

    attack_precision = (in_preds > 0.5).sum() / (
            (in_preds > 0.5).sum() + (out_preds > 0.5).sum()
    )

    # Compare to a baseline that merely guesses correct classified instances
    # are in and incorrectly classified instances are out.
    baseline_true_positives = \
        (y_pred_in.argmax(axis=1) == data.labels_train).mean()
    baseline_true_negatives = \
        (y_pred_out.argmax(axis=1) != data.labels_test).mean()
    baseline_attack_acc = \
        (baseline_true_positives + baseline_true_negatives) / 2.

    baseline_precision = \
        (y_pred_in.argmax(axis=1) == data.labels_train).sum() / (
                (y_pred_in.argmax(axis=1) == data.labels_train).sum() +
                (y_pred_out.argmax(axis=1) == data.labels_test).sum()
        )

    tqdm.write(
        f"\nTrue positive rate: {true_positives:0.4f}, " +
        f"true negative rate: {true_negatives:0.4f}"
    )
    tqdm.write(
        f"Shadow Attack Accuracy: {attack_acc:0.4f}, precision: {attack_precision:0.4f} " +
        f"(baseline: {baseline_attack_acc:0.4f}, {baseline_precision:0.4f})"
    )
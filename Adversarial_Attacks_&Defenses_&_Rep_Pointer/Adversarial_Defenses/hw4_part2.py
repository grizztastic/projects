# hw4_part2.py

import tqdm
import numpy as np
import tensorflow as tf

import hw4_utils
from hw4_utils import Splits

from hw4_mnist import HW4Model, MNISTModel
from hw4_part1 import Attacker, PGDAttacker

from sklearn.model_selection import train_test_split

class FineTunable(object):
    def __init__(self, finetune: bool = False):
        self.finetune = finetune

    def defend(self) -> None:
        if not self.finetune:
            # If we are not finetuning, we are training from scratch.
            self.model.build() # resets the model


class Defender(object):
    def __init__(
        self,
        attacker: Attacker,
        model: HW4Model,
        batch_size: int = 16,
        epochs: int = 2,
    ) -> None:

        self.attacker = attacker
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs

    def defend(self) -> None:
        pass


class AugmentDefender(Defender, FineTunable):
    def __init__(
        self,
        finetune: bool = False,
        *argv, **kwargs
    ) -> None:
        """
            finetune: bool -- finetune the existing model instead of training
              from scratch
        """
        Defender.__init__(self, *argv, **kwargs)
        FineTunable.__init__(self, finetune)

    def defend(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        augment_ratio: float = 0.1,
    ):
        """Defend by augmenting the training data.

        inputs:
          X: np.ndarray - input images [N, 28, 28, 1]
          Y: np.ndarray - ground truth classes [N]

          augment_ratio: float -- how much adversarial data to use as a ratio
            of training data

        returns Xadv, Yadv (the adversarial instances generated in defense),
        self.model should be defended
          Xadv: np.ndarray [N*augment_ratio, 28, 28, 1]
          Yadv: np.ndarray [N*augment_ratio]

        """

        Xadv = None # the adversarial instances generated in the process of
                    # defense
        Yadv = None # and their (correct) class

        # >>> Your code here <<<

        # Generate | X | * augment_ratio adversarial examples,
        adversarial = self.attacker.attack(X, Y)  # create adversarial images
        _, Xadv, _, Yadv = train_test_split(adversarial, Y, test_size=0.1)  # Generate | X | * augment_ratio adversarial examples
        Yadv = np.asmatrix(Yadv).T  # used to allow for proper augmentation to original data below
        Y = np.asmatrix(Y).T  # used to allow for proper augmentation to original data below
        FineTunable.defend(self)  # Resets model if not finetuning. If not finetuning,
        # make sure you generate the adversarial examples before you call this.
        # If finetuning, train on the adversarial examples.
        if self.finetune == True:
            self.model.train(X=Xadv, Y=Yadv, epochs=self.epochs,
                             batch_size=self.batch_size)  # train model on only adversarial images. I messed around with num epochs and batch size
        else:  # Otherwise retrain model with X, Y and the adversarial set
            X_combined = np.vstack((X, Xadv))  # augment adversarial images to original images
            Y_combined = np.vstack((Y, Yadv))  # augment adversairial image labels to original ground truth labels vector
            self.model.train(X=X_combined, Y=Y_combined, epochs=self.epochs,
                             batch_size=self.batch_size)  # train model on only adversarial images. I messed around with num epochs and batch size

        # Resets model if not finetuning. If not finetuning,
        # make sure you generate the adversarial examples
        # before you call this.

        # >>> End of your code <<<

        return Xadv, Yadv


class PreMadryDefender(Defender, FineTunable):
    def __init__(self, finetune: bool = False, *argv, **kwargs) -> None:
        Defender.__init__(self, *argv, **kwargs)
        FineTunable.__init__(self, finetune)

    def defend(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ):
        """
        Defend by augmenting the training data.

        inputs:
          X: np.ndarray - input images [N, 28, 28, 1]
          Y: np.ndarray - ground truth classes [N]

        returns Xadv, Yadv (the adversarial instances generated in defense),
        self.model should be defended
          Xadv: np.ndarray [N*augment_ratio, 28, 28, 1]
          Yadv: np.ndarray [N*augment_ratio]
        """

        FineTunable.defend(self) # resets model if not finetuning

        Xadv = None # the adversarial instances generated in the process of
                    # defense
        Yadv = None # and their (correct) class

        # >>> Your code here <<<

        # For each input batch, generate adversarial examples and train on them
        # instead of original data.
        '''Batch code taken from hw1 solutions'''
        n, h, w, c = X.shape
        n_class = Y.shape[0]
        n_batch = int(n // self.batch_size)

        X = X[:self.batch_size * n_batch]
        Y = Y[:self.batch_size * n_batch]
        print(Y.shape)

        X_batched = np.reshape(X, (n_batch, self.batch_size, h, w, c)) #put x into batches
        Y_batched = np.reshape(Y, (n_batch, len(Y)//self.batch_size)) #put y into batches
        Xadv = np.zeros((1, 28, 28, 1)) #create zero array to augment data to
        Yadv = np.zeros((1)) #create zero array to augment data to
        FineTunable.defend(self)  # resets model if not finetuning
        for epoch in range(self.epochs):
            for batch_idx in range(n_batch):
                X_batch_adv = self.attacker.attack(X_batched[batch_idx], Y_batched[batch_idx]) #generate adversarial examples for the batch
                self.model.train_on_batch(X=X_batch_adv, Y=Y_batched[batch_idx]) #train on batch function added to mnist model
                Xadv = np.vstack((Xadv, X_batch_adv)) #add adversarial images to the array
                Yadv = np.vstack((Yadv, np.asmatrix(Y_batched[batch_idx]).T)) #add ground truth labels to array
        Xadv = np.delete(Xadv, (0), axis=0) #delete first row
        Yadv = np.delete(Yadv, (0), axis=0) #delete first row
        # >>> End of your code <<<

        return Xadv, Yadv


### The rest of this file is for a BONUS exercise.


class MNISTModelSymbolic(MNISTModel):
    def __init__(self, *argv, **kwargs) -> None:
        super().__init__(*argv, **kwargs)

    def build(self, input: tf.Tensor) -> None:
        # Now takes input as a tensor that might be the result of an attack.

        pass


class PGDAttackerSymbolic(PGDAttacker):
    def __init__(
        self,
        model: MNISTModelSymbolic,
        *argv, **kwargs
    ) -> None:
        super().__init__(model, *argv, **kwargs)

    def symbolic_attack(
        self,
        X: tf.Tensor,
        Y: tf.Tensor
    ) -> tf.Tensor:
        """ Symbolic attack. For BONUS exercise in Part II. """
        pass


class MadryDefender(Defender):
    def __init__(
            self,
            model: MNISTModelSymbolic,
            attacker: PGDAttackerSymbolic,
            *argv, **kwargs) -> None:
        super().__init__(model=model, attacker=attacker, *argv, **kwargs)

    def defend(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        """
        Defend by augmenting the training data.

        inputs:
          X: np.ndarray - input images [N, 28, 28, 1]
          Y: np.ndarray - ground truth classes [N]

        returns nothing; self.model should have been defended
        """

        super().defend() # resets model if not finetuning

        # >>> Your code here <<<

        # BONUS: Implement Madry's defense. You will need to extend/adjust the
        # model class and the attacker class to make the proper symbolic
        # connections.

        # >>> End of your code <<<

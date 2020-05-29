from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.constraints import maxnorm
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np

#Load the training, validation, and test data to be used for training and predictions
x_train = np.load("train_X.npy").astype('float32')/255.0
y_train = np.load("train_y.npy").astype('float32')
onehot_train_Y = to_categorical(y_train)
val_X = np.load("val_X.npy").astype('float32')/255.0
val_Y = np.load("val_y.npy").astype('float32')
onehot_val_Y = to_categorical(val_Y)
private_test_X = np.load("private_test_X.npy").astype('float32')/255.0

#Build the model. Used sequential method and added layer by layer
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3),kernel_initializer='he_uniform', activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(128, (3, 3), activation='relu',kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu',kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))


epochs = 30
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #Compile model
model.summary()
model.fit(x_train, onehot_train_Y, validation_data=(val_X, onehot_val_Y), epochs=epochs, batch_size=128) #Fit the model
x = model.predict_classes(private_test_X) #Make predictions based on model
def save_csv(x, filename="submission.csv"):
    """save_csv Save the input into csv file

    Arguments:
        x {np.ndarray} -- input array

    Keyword Arguments:
        filename {str} -- The file name (default: {"submission.csv"})

    Raises:
        ValueError: Input data structure is not np.ndarray
    """
    if isinstance(x, np.ndarray):
        x = x.flatten()
        np.savetxt(filename, x, delimiter=',',fmt='%i')
    else:
        raise ValueError("The input is not an np.ndarray")
save_csv(x)

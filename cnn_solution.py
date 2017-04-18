"""
Demonstrating convolution neural networks
using Keras with a TensorFlow backend. Keras
is a high level machine learning package
which supports convolution, recurrent, and
standard neural networks, as well as allowing
you to define your own layer.

Edited by Vivek Karunakaran
"""
import os
import numpy as np
import pickle
import gzip
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from tensorflow.examples.tutorials.mnist import input_data
from numpy import array

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 3
epochs = 15

train_ex = 7000
img_h, img_w = 32, 32
def unpickle(file):
    """Load data"""
    import pickle
    with open(file, 'rb') as source:
        ret_dict = pickle.load(source, encoding='bytes')
    return ret_dict

def get_data():
    """
    Loads the data in.
    """
    tmp = unpickle("CIFAR-3.pickle")
    x_train = tmp['x'][:train_ex]
    x_train = x_train.reshape(x_train.shape[0], img_h, img_w, 1)
    x_train /= 255
    y_train = tmp['y'][:train_ex]

    x_tune = tmp['x'][7000:8000]
    x_tune = x_tune.reshape(x_tune.shape[0], img_h, img_w, 1)
    x_tune /= 255
    y_tune = tmp['y'][7000:8000]
	
    x_test = tmp['x'][8000:]
    x_test = x_test.reshape(x_test.shape[0], img_h, img_w, 1)
    x_test /= 255
    y_test = tmp['y'][8000:]

    return x_train, y_train,x_tune,y_tune, x_test, y_test

def convolution():
    """
    Keras follows the layers principle, where each layer
    is independent and can be stacked and merged together.
    The Sequential model assumes that there is one long
    stack, with no branching.
    """
    x_train, y_train, x_tune, y_tune, x_test, y_test = get_data()

    model = Sequential()

    """
    filters gives us the number of filters in the layer,the
    more filters we have, the more information we can learn

    kernel_size is the size of the convolution filter

    activation is the activation function on each node,
    we use relu, could also use sigmoid

    input_shape is the shape of the image. We reshaped
    the data above to get it in the right shape. The 1
    represents a grayscale image. If you had a colour
    image (RGB), the last dimension would be 3.
    """
    model.add(Conv2D(filters=30, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(img_h, img_w, 1)))
    
    """
    MaxPooling takes an NxM rectangle and find the maxiumum
    value in that square, and discards the rest. Since we are
    doing 2x2 pooling, it has the effect of halving the height
    and width of the image.
    """
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Sets a random 25% of nodes to 0 to prevent overfitting
    model.add(Dropout(0.25))

    # Note we don't need to give the shape between the first and
    # second layer, Keras figures that out for us.
    # second layer, Keras figures that out for us.
    model.add(Conv2D(28, kernel_size=(2, 2),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.20))


    model.add(Conv2D(28, kernel_size=(2, 2),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.20))


    model.add(Conv2D(28, kernel_size=(2, 2),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Before we had 6x6x32, now we have a flat 1152
    model.add(Flatten())

    # your standard fully connected NN layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    # Stochastic Gradient Descent
    sgd = SGD(lr=0.01, momentum=0.9)
    es = EarlyStopping(monitor='val_loss',
                       patience=5,  # epochs to wait after min loss
                       min_delta=0.0001)  # anything less than this counts as no change

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_tune,y_tune),
              callbacks=[es])

    score = model.evaluate(x_test, y_test)
    model.save_weights('model.hdf5')
    print('Test loss: {0}'.format(score[0]))
    print('Test accuracy: {0}'.format(score[1]))


    plt.figure('Predictions on MNIST', facecolor='gray')
    plt.set_cmap('gray')

    predictions = model.predict(x_test, verbose=0)

    for i in range(3):
        subplt = plt.subplot(int(i / 3) + 1, 3, i + 1)
        # no sense in showing labels if they don't match the letter
        hot_index = (predictions[i])
        subplt.set_title('Prediction: {0}'.format(hot_index))
        subplt.axis('off')
        letter = x_test[i]
        subplt.matshow(np.reshape(letter, [img_h, img_w]))
        plt.draw()
        # plt.savefig('cnn.png')

if __name__ == '__main__':
    convolution()
    plt.show()
    #os.system('open cnn.png')




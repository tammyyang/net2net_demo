from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os.path
import argparse
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from net2net import Net2Net


img_rows, img_cols = 28, 28
# number of classes
nb_classes = 10
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


def prepare_mnist_data():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, X_test, Y_train, Y_test


def create_model(insert=None):
    model = Sequential()

    layers = [Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)),
              Activation('relu'),
              Convolution2D(nb_filters, nb_conv, nb_conv),
              Activation('relu'),
              MaxPooling2D(pool_size=(nb_pool, nb_pool)),
              Dropout(0.25),
              Flatten(),
              Dense(128),
              Activation('relu'),
              Dropout(0.5),
              Dense(nb_classes),
              Activation('softmax')]

    if insert is not None:
        layers.insert(insert[0], insert[1])

    for layer in layers:
        model.add(layer)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Demo Net2Net on MNIST dataset"
        )

    parser.add_argument(
        "-r", "--retrain", default=False, action='store_true',
        help="To re-train and generate mnist_cnn.h5"
        )
    parser.add_argument(
        "--loss", default="categorical_crossentropy", type=str,
        help="Define loss function (default: categorical_crossentropy)"
        )
    parser.add_argument(
        "--optimizer", default="adadelta", type=str,
        help="Define optimizer (default: adadelta)"
        )
    parser.add_argument(
        "--weights", default="./mnist_cnn.h5", type=str,
        help="Path to the weight file (default: ./mnist_cnn.h5)"
        )
    parser.add_argument(
        "--batch-size", type=int, default=128, dest='size',
        help="Define batch size (default: 128)."
    )
    parser.add_argument(
        "--epochs", type=int, default=12,
        help="Define number of epochs (default: 12)."
    )

    args = parser.parse_args()

    n2n = Net2Net()
    new_layer = (7, Dense(128))
    X_train, X_test, Y_train, Y_test = prepare_mnist_data()

    ori_model = create_model()
    if args.retrain or not os.path.exists(args.weights):
        print('Training the original model and save weights to %s'
              % args.weights)
        ori_model.compile(loss=args.loss,
                          optimizer=args.optimizer,
                          metrics=['accuracy'])

        ori_model.fit(
            X_train, Y_train, batch_size=args.size, nb_epoch=args.epochs,
            verbose=1, validation_data=(X_test, Y_test))
        ori_model.save_weights(args.weights)

    ori_model.load_weights(args.weights)
    ori_layers = ori_model.layers

    model = create_model(insert=new_layer)
    model.summary()
    parms = ori_layers[new_layer[0]].get_weights()
    weights = parms[0]
    bias = parms[1]
    print("Net2Net: Original weights and bias of layer %i" % new_layer[0])
    print(weights.shape, bias.shape)
    new_w, new_b = n2n.deeper(weights, True)
    print("Net2Net: New weights and bias of layer %i" % (new_layer[0]+1))
    print(new_w.shape, new_b.shape)

    for j in range(0, len(ori_layers)):
        if j <= new_layer[0]:
            parm = ori_layers[j].get_weights()
            model.layers[j].set_weights(parm)
        elif j == new_layer[0] + 1:
            model.layers[j].set_weights([new_w, new_b])
        else:
            parm = ori_layers[j-1].get_weights()
            model.layers[j].set_weights(parm)

    model.compile(loss=args.loss,
                  optimizer=args.optimizer,
                  metrics=['accuracy'])

    X_train, X_test, Y_train, Y_test = prepare_mnist_data()
    model.fit(X_train, Y_train, batch_size=args.size, nb_epoch=args.epochs,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    main()

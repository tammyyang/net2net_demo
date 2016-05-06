'''
Replace NEWLAYERS with different layer you would like to use to expand
your network.

Parameters:
    insert_pos - index of the newly inserted layer
    layers     - a list of layers to be inserted

Example to insert a FC layer:
    NEWLAYERS = {'insert_pos': 8,
                 'layers': [Dense(128)]}
Example to insert a Conv layer:
    NEWLAYERS = {'insert_pos': 2,
                 'layers': [Activation('relu'),
                            Convolution2D(NB_FILTERS, NB_CONV, NB_CONV),
                            ZeroPadding2D((1, 1))]}

Padding is required to keep the size of the convolutional layers the same
before and after the expansion.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os.path
import argparse
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from net2net import Net2Net


IMGROWS, IMGCOLS = 28, 28
NB_CLASSES = 10
NB_FILTERS = 32
NB_POOL = 2
NB_CONV = 3

NEWLAYERS = {'insert_pos': 2,
             'layers': [Activation('relu'),
                        Convolution2D(NB_FILTERS, NB_CONV, NB_CONV),
                        ZeroPadding2D((1, 1))]}


def prepare_mnist_data():
    '''Ger MNIST data'''

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, IMGROWS, IMGCOLS)
    X_test = X_test.reshape(X_test.shape[0], 1, IMGROWS, IMGCOLS)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    return X_train, X_test, Y_train, Y_test


def create_model(insert=None):
    '''Create the basic model'''

    model = Sequential()

    layers = [Convolution2D(NB_FILTERS, NB_CONV, NB_CONV,
                            border_mode='valid',
                            input_shape=(1, IMGROWS, IMGCOLS)),
              Activation('relu'),
              MaxPooling2D(pool_size=(NB_POOL, NB_POOL)),
              Dropout(0.25),
              Flatten(),
              Dense(128),
              Activation('relu'),
              Dropout(0.5),
              Dense(NB_CLASSES),
              Activation('softmax')]

    if insert is not None:
        for l in insert['layers']:
            layers.insert(insert['insert_pos'], l)

    for layer in layers:
        model.add(layer)

    return model


def is_dense(layer):
    '''Check if the layer is dense (fully connected)'''

    ltype = layer.get_config()['name'].split('_')[0]
    if ltype == 'dense':
        return True
    return False


def is_convolutional(layer):
    '''Check if the layer is convolutional'''

    ltype = layer.get_config()['name'].split('_')[0]
    if ltype.find('convolution') > -1:
        return True
    return False


def find_ref_layer_idx(layers):
    '''
    Find the index of the reference layer. It looks for Conv or FC
    layer from (insert_pos - 1) to 0 of the ori_layers list and return
    the index of the found layer
    '''

    insert_pos = NEWLAYERS['insert_pos']
    for i in range(1, insert_pos + 1):
        ref_layer = layers[insert_pos - i]
        if is_convolutional(ref_layer) or is_dense(ref_layer):
            return insert_pos - i


def find_major_layer_idx():
    '''Looking for the Conv or FC layer in NEWLAYERS['layers']'''

    for i in range(0, len(NEWLAYERS['layers'])):
        layer = NEWLAYERS['layers'][i]
        if is_convolutional(layer) or is_dense(layer):
            return i
    return -1


def get_deeper_weights(ref_layer):
    '''
    To calculate new weights to make the net deeper using Net2Net class,
    one needs to swap the axes for the right order.
    Dim of Keras conv layer: (OutChannel, InChannel, kH, kW)
           conv layer Net2Net class accepts: (kH, kW, InChannel, OutChannel)
    '''
    parms = ref_layer.get_weights()
    n2n = Net2Net()
    if is_convolutional(ref_layer):
        weights = parms[0].swapaxes(0, 2).swapaxes(1, 3).swapaxes(2, 3)
        new_w, new_b = n2n.deeper(weights, True)
        new_w = new_w.swapaxes(0, 2).swapaxes(1, 3)
    else:
        weights = parms[0]
        new_w, new_b = n2n.deeper(weights, True)
    return new_w, new_b


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

    X_train, X_test, Y_train, Y_test = prepare_mnist_data()

    ori_model = create_model()
    ori_model.summary()
    if args.retrain or not os.path.exists(args.weights):
        print('Training the original model and save weights to %s'
              % args.weights)
        ori_model.compile(loss=args.loss,
                          optimizer=args.optimizer,
                          metrics=['accuracy'])

        ori_model.fit(
            X_train, Y_train, batch_size=args.size, nb_epoch=args.epochs,
            verbose=1, validation_data=(X_test, Y_test))
        ori_model.save_weights(args.weights, overwrite=True)

    ori_model.load_weights(args.weights)
    ori_layers = ori_model.layers

    model = create_model(insert=NEWLAYERS)
    model.summary()
    i = find_ref_layer_idx(ori_layers)

    # Layers such as ZeroPadding2D or Activation gets no weights
    shift = find_major_layer_idx()
    shift = 0 if shift < 0 else shift

    new_w, new_b = get_deeper_weights(ori_layers[i])

    n_new_layers = len(NEWLAYERS['layers'])
    for j in range(0, len(ori_layers)):
        if j <= i:
            parm = ori_layers[j].get_weights()
            model.layers[j].set_weights(parm)
        elif j == i + NEWLAYERS['insert_pos'] + shift:
            model.layers[j].set_weights([new_w, new_b])
        elif j > i + n_new_layers:
            parm = ori_layers[j - n_new_layers].get_weights()
            model.layers[j].set_weights(parm)

    model.compile(loss=args.loss,
                  optimizer=args.optimizer,
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=args.size, nb_epoch=args.epochs,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    main()

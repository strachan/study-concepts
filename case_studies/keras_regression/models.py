from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model


def create_mlp(dim, regress=False):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation='relu'))
    model.add(Dense(4, activation='relu'))

    if regress:
        model.add(Dense(1, activation='linear'))

    return model


def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):

    input_shape = (height, width, depth)
    chan_dim = -1

    inputs = Input(shape=input_shape)

    for(i, f) in enumerate(filters):
        if i == 0:
            x = inputs

        x = Conv2D(f, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=chan_dim)(x)
    x = Dropout(0.5)(x)

    x = Dense(4)(x)
    x = Activation('relu')(x)

    if regress:
        x = Dense(1, activation='linear')(x)

    model = Model(inputs, x)

    return model

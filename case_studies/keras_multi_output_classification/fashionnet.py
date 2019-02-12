from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf


class FashionNet:

    @staticmethod
    def build_category_branch(inputs, num_categories, final_act='softmax',
                              chan_dim=-1):
        # convert the RGB to grayscale representation to focus on the actual
        # structural components in the image, ensuring the network does not
        # learn to jointly associate a particular color with a clothing type
        x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)

        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        # building redundancy and avoiding overfitting
        x = Dropout(0.25)(x)

        for filter_size in [64, 128]:
            x = Conv2D(filter_size, (3, 3), padding='same')(x)
            x = Activation('relu')(x)
            x = BatchNormalization(axis=chan_dim)(x)
            x = Conv2D(filter_size, (3, 3), padding='same')(x)
            x = Activation('relu')(x)
            x = BatchNormalization(axis=chan_dim)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(256)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_categories)(x)
        x = Activation(final_act, name='category_output')(x)

        return x

    @staticmethod
    def build_color_branch(inputs, num_colors, final_act='softmax',
                           chan_dim=-1):
        x = Conv2D(16, (3, 3), padding='same')(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        for _ in range(2):
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = Activation('relu')(x)
            x = BatchNormalization(axis=chan_dim)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_colors)(x)
        x = Activation(final_act, name='color_output')(x)

        return x

    @staticmethod
    def build(width, height, num_categories, num_colors, final_act='softmax'):

        input_shape = (height, width, 3)
        chan_dim = -1

        inputs = Input(shape=input_shape)
        category_branch = FashionNet.build_category_branch(inputs,
                                                           num_categories,
                                                           final_act=final_act,
                                                           chan_dim=chan_dim)
        color_branch = FashionNet.build_color_branch(inputs,
                                                     num_colors,
                                                     final_act=final_act,
                                                     chan_dim=chan_dim)

        model = Model(inputs=inputs, outputs=[category_branch, color_branch],
                      name='fashionnet')

        return model

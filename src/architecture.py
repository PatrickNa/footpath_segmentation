#!/usr/bin/env python3

from tensorflow.keras import Model
from tensorflow.keras.layers import (Add, Conv2D, Conv2DTranspose, Cropping2D,
                                     Dropout, Input, MaxPool2D, Softmax,
                                     ZeroPadding2D)
from tensorflow.keras.regularizers import l2


class FootpathModelArchitecture():
    """Defines the architecture of the used model.

    Args:
        input_shape (int tuple): Defines the dimensions of the data fed to the 
                                    model.
    """

    def __init__(self, input_shape):
        self.inputs = Input(shape=input_shape, name='inputs')
        self.conv1_1 = Conv2D(filters=64, kernel_size=3, strides=(1, 1),
            padding='same', activation='relu', name='conv1_1')(self.inputs)
        self.conv1_2 = Conv2D(filters=64, kernel_size=3, strides=(1, 1),
            padding='same', activation='relu', name='conv1_2')(self.conv1_1)
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
            padding='same', name='pool1')(self.conv1_2)

        self.conv2_1 = Conv2D(filters=128, kernel_size=3, strides=(1, 1),
            padding='same', activation='relu', name='conv2_1')(self.pool1)
        self.conv2_2 = Conv2D(filters=128, kernel_size=3, strides=(1, 1),
            padding='same', activation='relu', name='conv2_2')(self.conv2_1)
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
            padding='same', name='pool2')(self.conv2_2)

        self.conv3_1 = Conv2D(filters=256, kernel_size=3, strides=(1, 1),
            padding='same', activation='relu', name='conv3_1')(self.pool2)
        self.conv3_2 = Conv2D(filters=256, kernel_size=3, strides=(1, 1),
            padding='same', activation='relu', name='conv3_2')(self.conv3_1)
        self.conv3_3 = Conv2D(filters=256, kernel_size=3, strides=(1, 1),
            padding='same', activation='relu', name='conv3_3')(self.conv3_2)
        self.pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
            padding='same', name='pool3')(self.conv3_3)

        self.conv4_1 = Conv2D(filters=512, kernel_size=3, strides=(1, 1),
            padding='same', activation='relu', name='conv4_1')(self.pool3)
        self.conv4_2 = Conv2D(filters=512, kernel_size=3, strides=(1, 1),
            padding='same', activation='relu', name='conv4_2')(self.conv4_1)
        self.conv4_3 = Conv2D(filters=512, kernel_size=3, strides=(1, 1),
            padding='same', activation='relu', name='conv4_3')(self.conv4_2)
        self.pool4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
            padding='same', name='pool4')(self.conv4_3)

        self.conv5_1 = Conv2D(filters=512, kernel_size=3, strides=(1, 1),
            padding='same', activation='relu', name='conv5_1')(self.pool4)
        self.conv5_2 = Conv2D(filters=512, kernel_size=3, strides=(1, 1),
            padding='same', activation='relu', name='conv5_2')(self.conv5_1)
        self.conv5_3 = Conv2D(filters=512, kernel_size=3, strides=(1, 1),
            padding='same', activation='relu', name='conv5_3')(self.conv5_2)
        self.pool5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
            padding='same', name='pool5')(self.conv5_3)

        self.fc6 = Conv2D(filters=4096, kernel_size=(7, 7), strides=(1, 1),
            padding='same', activation='relu', name='fc6')(self.pool5)
        self.fc6 = Dropout(rate=0.5, name='dropout_fc6')(self.fc6)

        self.fc7 = Conv2D(filters=4096, kernel_size=(1, 1), strides=(1, 1),
            padding='same', activation='relu', name='fc7')(self.fc6)
        self.fc7 = Dropout(rate=0.5, name='dropout_fc7')(self.fc7)

        self.score = Conv2D(filters=2, kernel_size=(1, 1), strides=(1, 1),
            kernel_regularizer=l2(l=(2/4096)**0.5), name='score')(self.fc7)

        self.upscore2 = Conv2DTranspose(filters=2, kernel_size=(4, 4),
            strides=(2, 2), padding='same', name='upscore2')(self.score)

        self.score_pool4 = Conv2D(filters=2, kernel_size=(1, 1), strides=(1, 1),
            kernel_regularizer=l2(l=0.001), name='score_pool4')(self.pool4)

        row_shape_diff = (self.upscore2.get_shape()[1] -
                          self.score_pool4.get_shape()[1])
        column_shape_diff = (self.upscore2.get_shape()[2] -
                             self.score_pool4.get_shape()[2])
        self.score_pool4 = ZeroPadding2D(padding=((0, row_shape_diff), (0, column_shape_diff)))(self.score_pool4)

        self.fuse_pool4 = Add(name='fuse_pool4')([self.upscore2, self.score_pool4])
        self.upscore4 = Conv2DTranspose(filters=2, kernel_size=(4, 4), strides=(
            2, 2), padding='same', name='upscore4')(self.fuse_pool4)
        self.score_pool3 = Conv2D(filters=2, kernel_size=(1, 1), strides=(
            1, 1),  padding='same', kernel_regularizer=l2(l=0.0001),
            name='score_pool3')(self.pool3)

        row_shape_diff = self.upscore4.get_shape()[1] - self.score_pool3.get_shape()[1]
        column_shape_diff = self.upscore4.get_shape()[2] - self.score_pool3.get_shape()[2]
        self.score_pool3 = ZeroPadding2D(padding=((0, row_shape_diff), (0, column_shape_diff)))(self.score_pool3)

        self.fuse_pool3 = Add(name='fuse_pool3')([self.upscore4, self.score_pool3])

        self.upscore32 = Conv2DTranspose(filters=2, kernel_size=(16, 16),
            strides=(8, 8), padding='same', name='upscore32')(self.fuse_pool3)

        row_shape_diff = self.upscore32.get_shape()[1] - self.inputs.get_shape()[1]
        column_shape_diff = self.upscore32.get_shape()[2] - self.inputs.get_shape()[2]
        self.upscore32 = Cropping2D(cropping=((0, row_shape_diff), (0, column_shape_diff)))(self.upscore32)

        self.final_conv = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),
            padding='same', activation='sigmoid', name='final_conv')(self.upscore32)

        self.footpath_model = Model(inputs=self.inputs,
            outputs=self.final_conv)
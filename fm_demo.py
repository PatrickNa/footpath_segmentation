#!/usr/bin/env python3
import os
import pathlib
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam

from src.architecture import SegmentationModel


def load_vgg16_weights(fm_model):

    keras_vgg16 = VGG16()

    MAX_POOL1 = 3
    MAX_POOL2 = 6
    MAX_POOL3 = 10
    MAX_POOL4 = 14
    max_pool_indices = [MAX_POOL1, MAX_POOL2, MAX_POOL3, MAX_POOL4]

    for i in range(1, 17):
        if i in max_pool_indices:
            continue
        else:
            weights = keras_vgg16.get_layer(index=i).get_weights()
            fm_model.get_layer(index=i).set_weights(weights)


def map_to_masks(frame_directory):
    frame = tf.io.read_file(frame_directory)
    frame = tf.image.decode_png(frame, channels=3)
    frame = tf.image.convert_image_dtype(frame, tf.uint8)

    mask_directory = tf.strings.regex_replace(
        input=frame_directory, pattern='frames', rewrite=('masks'))

    mask = tf.io.read_file(mask_directory)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)
    mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)

    return {'frame': frame, 'mask': mask}


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = (tf.cast(input_mask, tf.float32) - tf.math.reduce_min(input_mask)) / (tf.math.reduce_max(input_mask) - tf.math.reduce_min(input_mask))
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['frame'], (375, 1242))
    input_mask = tf.image.resize(datapoint['mask'], (375, 1242))

    if (tf.random.uniform(()) > 0.5):
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['frame'], (375, 1242))
    input_mask = tf.image.resize(datapoint['mask'], (375, 1242))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


if __name__ == "__main__":
    training_directory = '/opt/share/datasets/kitti_road_data/training/'

    LEARNING_RATE = 1e-05
    EPSILON = 1e-05

    input_shape = (375, 1242, 3)
    fm_model = SegmentationModel(input_shape).segmentation_model
    load_vgg16_weights(fm_model)

    fm_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE,
                     epsilon=EPSILON),
                     loss=BinaryCrossentropy(from_logits=True),
                     metrics=[MeanIoU(num_classes=2)])

    training_frames_directory = training_directory + 'training_frames'
    training_frames_directory = pathlib.Path(training_frames_directory)

    training_frames = tf.data.Dataset.list_files(
        str(training_frames_directory/'u*'))
    training_data = training_frames.map(map_to_masks)

    validation_frames_directory = training_directory + 'validation_frames'
    validation_frames_directory = pathlib.Path(validation_frames_directory)
    validation_frames = tf.data.Dataset.list_files(
        str(validation_frames_directory/'u*'))
    validation_data = validation_frames.map(map_to_masks)

    data_set = {'train': training_data, 'test': validation_data}

    TRAIN_LENGTH = len(list(training_frames_directory.glob('**/*.png')))
    BATCH_SIZE = 1
    EPOCHS = 12000
    VALIDATION_STEPS = 100
    SAVE_CHECKPOINT_STEPS = TRAIN_LENGTH * 20

    BUFFER_SIZE = TRAIN_LENGTH + 1

    train = data_set['train'].map(load_image_train, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = data_set['test'].map(load_image_test)

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test.batch(BATCH_SIZE).repeat()

    checkpoint_path = "fm_model_checkpoints/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_freq=SAVE_CHECKPOINT_STEPS)

    log_dir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    fm_model.fit(train_dataset, 
                 epochs=EPOCHS,
                 steps_per_epoch=TRAIN_LENGTH,
                 validation_steps=VALIDATION_STEPS,
                 validation_data=test_dataset,
                 callbacks=[checkpoint_callback, tensorboard_callback])
    fm_model.save('fm_model.h5')

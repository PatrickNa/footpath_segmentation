#!/usr/bin/env python3

import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


def load_vgg16_weights(fm_model):
    """Load the weights of VGG16 up to conv5_3 of the passed model.

    The weights of the first five convolutional layers of model are replaced
    by the weights of the VGG16 model available in tensorflow Keras. This is
    especially useful when the model is trained for the first time.

    Args:
        fm_model (tensorflow keras model): The model of which the weights shall
                                            be replaced by the VGG16 ones.
    """
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


def map_to_masks(image_directory, pattern):
    """Brings images with their corresponding masks together.

    This function is applied in the dataset preparation process. Images are
    mapped to their corresponding mask files. The pattern argument describes
    which element of the directory needs to be replaced by what in order to
    find the mask of an image. E.g. ~/train_images/123.png and 
    pattern=('images', 'masks') would result in ~/train_masks/123.png

    Args:
        image_directory (string): A string which points to the directory where
                                    images are located that shall be matched to
                                    corresponding masks.
        pattern (string tuple): Consists of two elements which describe the
                                    parts of the image directory that need to
                                    be replaced to find the corresponding mask.

    Returns:
        A dictionary containing the image and the corresponding mask as tensors.
    """
    image = tf.io.read_file(image_directory)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_directory = tf.strings.regex_replace(input=image_directory,
    pattern=pattern[0], rewrite=pattern[1])
    
    mask = tf.io.read_file(mask_directory)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)
    mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
    
    return {'image': image, 'mask': mask}


def normalize(input_image, input_mask):
    """Normalize image and corresponding mask.

    The image is normalized between zero and one. The corresponding mask,
    due to the nature of masks consisting of only two classes, is transformed
    to values either zero or one.

    Args:
        input_image: The original image.
        input_mask: The corresponding mask of the image, consisting of two 
                        classes - footpath/street and background.

    Returns:
        The normalized image and mask are returned. 
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = ((tf.cast(input_mask, tf.float32) -
                    tf.math.reduce_min(input_mask)) /
                    (tf.math.reduce_max(input_mask) -
                    tf.math.reduce_min(input_mask)))
    return input_image, input_mask


@tf.function
def load_image_train(datapoint, image_dimensions):
    """Apply augmentation, scaling and normalization to training image and mask.

    This function is applied to the training dataset in order to augment the 
    data, resize the images to the appropriate dimensions and normalize the 
    values.

    Args:
        datapoint (dict): A datapoint consists of the original image and 
                            corresponding mask.
        image_dimension (int tuple): The width and height of the input 
                                        considered in the model of interest.

    Returns:
        The image and corresponding mask with the applied changes described
        above.  
    """
    input_image = tf.image.resize(datapoint['image'], image_dimensions)
    input_mask = tf.image.resize(datapoint['mask'], image_dimensions)

    if (tf.random.uniform(()) > 0.5):
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint, image_dimensions):
    """Apply scaling and normalization to validation image and mask.

    This function is applied to the validation dataset in order to resize the
    dimensions and apply normalization similar of the training dataset.

    Args:
        datapoint (dict): A datapoint consists of the original image and 
                            corresponding mask.
        image_dimension (int tuple): The width and height of the input 
                                        considered in the model of interest.

    Returns:
        The image and corresponding mask with the applied changes described
        above. 
    """
    input_image = tf.image.resize(datapoint['image'], image_dimensions)
    input_mask = tf.image.resize(datapoint['mask'], image_dimensions)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def create_training_dataset(training_data_directory, 
                            training_image_subfolder='training_images', 
                            validation_image_subfolder='validation_images', 
                            pattern=('images', 'masks')):
    """Create a dictionary consisting of training and validation datasets.

    This function creates tensorflow datasets for training and validation, 
    each consisting of images and corresponding masks.

    Args:
        training_data_directory (string): A string which points to the directory
                                            in which the training and validation
                                            data are placed.
        training_image_subfolder (string): The name of the subfolder which
                                                contains the training images.
        validation_image_subfolder (string): The ame of the subfolder which
                                                contains the validation images.
        pattern (string tuple): The first element describes the part of the
                                    folder name which shall be replace by the
                                    second element in order to finde the 
                                    corresponding mask of an image.

    Returns:
        A dictionary with training and validation tensorflow datasets. 
    """
    
    training_image_path = pathlib.Path(training_data_directory +
                                       training_image_subfolder)
    training_images = tf.data.Dataset.list_files(str(training_image_path/'*'))
    training_data = training_images.map(lambda training_images: map_to_masks(training_images, pattern))

    validation_image_path  = pathlib.Path(training_data_directory + validation_image_subfolder)
    validation_images = tf.data.Dataset.list_files(str(validation_image_path/'*'))
    validation_data = validation_images.map(lambda validation_images: map_to_masks(validation_images, pattern))

    return {'training':training_data, 'validation':validation_data}

def sum_up_training_dataset(dataset, buffer_size, batch_size, repeat=True, number_of_repetitions=0):
    """Define how the training dataset is suppose to behave during training.

    This function is applied to the training dataset just before the actual
    training process. The characteristics defined here address how images are
    picked and how large the batch size is.

    Args:
        dataset (tensorflow dataset): The dataset to which the functions are 
                                        applied.
        buffer_size (int): The number of elements that should be considered in
                            shuffeling process.
        batch_size (int): Defines the number of images per step in an epoch 
                            which shall be considered.
        repeat (boolean): If set to false the training data is only considered 
                            once. If set to true, the dataset is either 
                            considered endlessly or number_of_repetitions times.
        number_of_repetitions (int): Defines how often the training data is 
                                        considered before the training process 
                                        stops. If set to anything smaller or 
                                        equal to zero the termination of the
                                        training process is defined by another
                                        parameter and the training data is 
                                        considered endlessly.

    Returns:
        The tensorflow dataset which the applied changes described above. 
    """
    if (repeat):
        if (number_of_repetitions > 0):
            dataset = dataset.cache().shuffle(buffer_size).batch(batch_size).repeat(number_of_repetitions)
        else:
            dataset = dataset.cache().shuffle(buffer_size).batch(batch_size).repeat()
    else:
        dataset = dataset.cache().shuffle(buffer_size).batch(batch_size)
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def sum_up_validation_dataset(dataset, batch_size, repeat=True, 
                              number_of_repetitions=0):
    """Define how the validation dataset is suppose to behave during training.

    This function is applied to the validation dataset just before the actual
    training process. The characteristics defined here address how images are
    picked and how large the batch size is.

    Args:
        dataset (tensorflow dataset): The dataset to which the functions are 
                                        applied.
        batch_size (int): Defines the number of images per validation step.
        repeat (boolean): If set to false the validation data is only considered 
                            once. If set to true, the dataset is either 
                            considered endlessly or number_of_repetitions times.
        number_of_repetitions (int): Defines how often the validation data is 
                                        considered.

    Returns:
        The tensorflow dataset which the applied changes described above. 
    """
    if (repeat):
        if (number_of_repetitions > 0):
            dataset = dataset.batch(batch_size).repeat(number_of_repetitions)
        else:
            dataset = dataset.batch(batch_size).repeat()
    else:
        dataset = dataset.batch(batch_size)
    return dataset


def create_checkpoint_callback(checkpoint_directory, save_best_only=True, monitor='binary_crossentropy',
                               save_weights_only=True, save_freq=1):
    """Create a checkpoint callback for the training process.

    This function is a wrapper for the native tensorflow ModelCheckpoint object.
    The parameters requested in this wrapper showed great importance
    in an early development stage. This function is suppose to emphasize their
    role an prevent to forget about them.

    Args:
        checkpoint_directory (string): The path to the directory where 
                                        checkpoints shall be saved during 
                                        training.
        save_weights (boolean): A flag that enables/ disables the saving of 
                                weights.
        save_freq (int): This argument defines how often checkpoints are 
                            created. Using datasets, this value refers to steps,
                            rather than epochs.

    Returns:
        A ModelCheckpoint callback object, which can be added to the training
        execution function.  
    """
    return ModelCheckpoint(filepath=checkpoint_directory, monitor=monitor, save_best_only=save_best_only,
                           save_weights_only=save_weights_only, verbose=1,
                           save_freq=save_freq)

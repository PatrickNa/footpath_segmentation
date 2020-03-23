#!/usr/bin/env python3
import json
import os
from random import randrange

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img


def load_random_image(data_directory, image_dimension):
    """Shuffles over a directory and returns a random image.

    This function can be used to randomly pick a test image to test the
    performance and output of the model.

    Args:
        data_directory (string): The directory as string from where the image 
                                    is returned.
        image_dimension (int tuple): The size in which the image is returned.

    Returns:
        An image as PIL image of the dimensions specified as function parameter.
    """
    list_of_filenames = os.listdir(data_directory)
    random_filename = list_of_filenames[randrange(start=0, 
                                            stop=(len(list_of_filenames)-1), 
                                            step=1)]
    image_path = data_directory + random_filename
    return load_img(path=image_path, target_size=image_dimension)


def preprocess_image(image, expand_dimension=False):
    """Preprocesses an image before it is predicted by a model.

    The changes applied to the image are the same changes that were applied
    to the training data. Thus once can expect the model to treat it the same
    way. If the the second parameter is set to true, the function adds a batch
    size of one to the image. This fourth dimension is needed by the model to
    apply predicition.

    Args:
        image (PIL or numpy array): The image which is supoose to be 
                                        preprocessed.
        expand_dimension (boolean): If set to true, the image receives a fourth
                                        dimension, a batch of size one.

    Returns:
        The function returns a preprocessed version of the input image.   
    """
    preprocessed_image = tf.cast(np.array(image), tf.float32) / 255.
    
    if (expand_dimension):
        preprocessed_image = tf.expand_dims(preprocessed_image, 0)

    return preprocessed_image

def save_predicted_mask(filepath, predicted_mask):
    """Save the output of a prediction as an image.

    This functions allows to save the prediction, the output of the model as
    PNG file to the disk. For that a few postprocessing actions need to be
    applied to the output.

    Args:
        filepath (string): The directory and filename.
        predicted_mask: The outcome of the model of shape (1, width, height, 1).
    """

    mask = np.squeeze(predicted_mask > 0.5)
    mask = np.uint8(np.stack((mask*0, mask*255, mask*0), -1))
    mask_as_image = Image.fromarray(mask)
    mask_as_image.save(filepath, 'PNG')


def create_summary_file(filepath, model, parameters, other=None):
    """Creates a file which contains information about the training/prediction
       process.

    This function creates a json file which summarizes with which parameters
    and architecture the user trained or predicted. It is especially useful when
    a lot of test runs with different datasets are made.

    Args:
        filepath (string): The directory and file name for the summary. 
        model (tf.keras.Model): The model which was used to train/predict.
        parameters (dict): The parameters that were loaded for the task.
        other (dict): Further custom information that the user wants to keep
                      track of, e.g. loss function.
    """
    model_summary_as_string = model.to_json(indent=4)
    model_summary_as_json = json.loads(model_summary_as_string)
    summary = {
        'model_summary': model_summary_as_json,
        'action_paramaters': parameters,
    }

    if (other != None):
        summary['other_parameters'] = other
    
    with open(filepath, 'w') as summary_file:
        json.dump(summary, summary_file, indent=4)

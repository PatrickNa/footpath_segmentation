#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from src.training_preparation import normalize

from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img


def create_overlay(input_image, mask, alpha=0.3, from_prediction=False):
    """Overlay the original image with the corresponding, predefined or 
        predicted mask.

    This function visualizes how accurate a mask is, putting a layer of green
    over the original image emphasizing what footpath/street is.

    Args:
        input_image (numpy array): The original RGB image.
        mask (numpy array): The corresponding mask as grayscale image.
        alpha (float): This value defines how transparent the overlay is 
                        suppose to be.
        from_prediction (boolean): If the mask is coming as prediction from a 
                                    model, it has an additional dimension and
                                    needs to be processed differently. This flag
                                    indicates how to proceed within the 
                                    function.

    Returns:
        A PIL image which shows the original image with an overlay of the 
        provided mask. 
    """
    if (from_prediction):
        mask = np.uint8((np.squeeze(mask) > 0.5) * 255)
        mask = np.stack((mask*0, mask, mask*0), -1)
        mask = Image.fromarray(np.uint8(mask))
    else:
        mask = np.squeeze(np.array(mask))
        if (len(mask.shape) < 3):
            mask = np.uint8(((mask - np.min(mask)) /
                            (np.max(mask) - np.min(mask))) * 255)
            mask = np.stack((mask*0, mask, mask*0), -1)
        mask = Image.fromarray(np.uint8(mask))
    
    input_image = np.array(input_image)
    if (len(input_image.shape) > 3):
        input_image = np.squeeze(input_image)
    if (np.max(input_image) <= 1):
        input_image *= 255

    original_image = Image.fromarray(np.uint8(input_image))
    return Image.blend(original_image, mask, alpha)


def display_images(image_list, label_list):
    """Display images and corresponding labels in a structured manner.

    This functions opens the images defined in the image_list parameter and 
    annotates the corresponding labels from label_list. Multiple images are 
    displayed in a row, allowing the user to easily compare them.

    Args:
        image_list (list of images): This list contains the images that shall
                                        be displayed.
        label_list (list of strings): Labels, describing the images at the same
                                        index of the image_list parameter.    
    """
    plt.figure(figsize=(15, 15))


    for i in range(len(image_list)):
        image = np.array(image_list[i])
        image = np.squeeze(image) if (len(image.shape) > 3) else image
        image = np.expand_dims(image, -1) if (len(image.shape) < 3) else image
        plt.subplot(1, len(image_list), i+1)
        plt.title(label_list[i])
        plt.imshow(array_to_img(image))
        plt.axis('off')
    plt.show()
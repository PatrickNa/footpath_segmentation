#!/usr/bin/env python3


def freeze_layers(model, indices):
    """Freezes the weights of layers so that they will not be changed anymore.

    This function freezes the weights of a model. In particular the weights of
    the layers of a model. The model and layer 
    information are passed as parameters. The indices point on 

    Args:
        model (tf.keras.Model): The model of interest is passed as a parameter.
        indices (int array): An array of all layer indices that shall be
                             freezed. If only one single, negative value is 
                             passed (e.g. -1) all layers up to this layer are
                             affected.
    """

    if (len(indices) == 1) and (indices[0] < 0):
        for layer in model.layers[:indices[0]]:
            layer.trainable = False
    else:
        for layer_index in indices:
            layer_of_interest = model.get_layer(index=layer_index)
            layer_of_interest.trainable = False

#!/usr/bin/env python3


def freeze_layers_by_index(model, indices):
    """Freezes the weights of layers so that they will not be changed anymore.

    This function freezes the weights of a model. In particular the weights of
    the layers of a model. The model and layer information are passed as 
    parameters. The indices point to the layers that will not be trainable 
    anymore.

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


def freeze_layers_by_name(model, names):
    """Freezes the weights of layers so that they will not be changed anymore.

    This function freezes the weights of a model. In particular the weights of
    the layers of a model. The model and layer information are passed as 
    parameters. The names determine the layers that will not be trainable 
    anymore.

    Args:
        model (tf.keras.Model): The model of interest is passed as a parameter.
        names (string array): An array of all layer names that shall be freezed.
    """

    for layer_name in names:
        layer_of_interest = model.get_layer(name=layer_name)
        layer_of_interest.trainable = False
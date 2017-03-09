import lasagne.layers as ll


def name_layers(layers, name, use_start_as_input=True):
    if use_start_as_input:
        layers_to_name = ll.get_all_layers(layers[-1], treat_as_input=[layers[0]])
    else:
        layers_to_name = ll.get_all_layers(layers[-1])

    for i, layer in enumerate(layers_to_name):
        this_name = name + '_' + str(i)
        layer.name = this_name
        params = layer.get_params()
        for param in params:
            param.name += '_' + this_name


def find_bn_updates(layers):
    """
        Return a list of tuples of all the bn_layers in this list of layers
    """
    bn_updates = []
    layers = ll.get_all_layers(layers)
    for layer in layers:
        if hasattr(layer, 'bn_updates'):
            bn_updates.extend(layer.bn_updates)
    return bn_updates

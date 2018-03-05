from deepmonster.nnet.blocks import ConvBlock
from deepmonster.nnet.convolution import MaxPooling
from deepmonster.nnet.extras import Flatten

def make_vgg(image_size, channels, topology='16a', last_max_pool=True,
             finish_with_flatten=True):
    """Begins a vgg architecture with ConvBlocks of the appropriate sizes. Do not forget that
    ConvBlocks have no activation or normalization defined and these can be setup latter
    through a Feedforward initialization.

    topology can have 2 formats:
        - str: maps to a known topology
        - list: first element == starting feature maps, followed by
        tuples:
            len(tuple) == nb of conv layer
            tuple[i] == kernel size of conv layer
            after each tuple a maxpooling which downsamples by 2 is inserted
        and optionnally ints:
            the feature map policy will be to double in size if no int is present
            else it will take the int as next feature map
    """
    known_topology = {
        '16a': [64, (3,3), (3,3), (3,3,1), (3,3,1), 512, (3,3,1)],
        '16b': [64, (3,3), (3,3), (3,3,3), (3,3,3), 512, (3,3,3)],
    }
    if isinstance(topology, str):
        topology = known_topology[topology]
    elif not isinstance(topology, list):
        raise TypeError("invalid topology")

    assert isinstance(topology[0], int), \
            "need at least first element to be an int (first feature maps size)"
    assert all(isinstance(i, (int, tuple)) for i in topology), \
            "invalid topology (elements should be int or tuple)"

    # fill in all gaps with int
    new_topology = []
    next_state = int
    last_int = None
    for t in topology:
        if isinstance(t, int) and next_state is int:
            new_topology.append(t)
            last_int = t
            next_state = tuple
        elif isinstance(t, tuple) and next_state is tuple:
            new_topology.append(t)
            next_state = int
        elif isinstance(t, tuple) and next_state is int:
            # double last_int value and insert it before insterting the tuple
            last_int *= 2
            new_topology.append(last_int)
            new_topology.append(t)

    layers = []
    c = None
    for i, el in enumerate(new_topology):
        if isinstance(el, int):
            c = el
            continue
        for t in el:
            assert t % 2 == 1
            if i == 1:
                layers.append(ConvBlock(t, c, num_channels=channels, image_size=image_size,
                                        padding='half'))
                continue
            layers.append(ConvBlock(t, c, padding='half'))

        if i < len(new_topology) - 1 or last_max_pool:
            layers.append(MaxPooling())

    if finish_with_flatten:
        layers.append(Flatten())

    return layers

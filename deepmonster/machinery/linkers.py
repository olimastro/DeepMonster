from deepmonster.nnet.utils import assert_iterable_return_iterable


class LinksHolder(object):
    """Knows every link that has been instanciated during a model's
    definition of its graph.
    """
    def __init__(self):
       self._links = []

    def __iter__(self):
        def linksholder_iterator():
            for link in self._links:
                yield link
        return linksholder_iterator()

    @property
    def links(self):
        return self._links

    def store_link(self, newlink):
        for link in self._links:
            # can we compare a link with a link of a diff class?
            if type(link) == type(newlink):
                link.sanity_check(newlink)

        self._links.append(newlink)

    def clear_links(self):
        print "WARNING: Links are being cleared, hope this was intended"
        self._links = []



class VariableLink(object):
    """Link for a model's variable. A typical
    use of these links would for example if the variables are needed
    for an extension or are a special cost in the bprop graph.

    Think of linkers has helpers for the many different usage a variables
    in the graph can have in the model and / or with training to pass them
    around to the objects that need them.
    """
    def __init__(self, var):
        # it would make no sense to have the same var more than once?
        self.var = assert_iterable_return_iterable(var, 'set')

    @property
    def model_var(self):
        # some var could be deep in a multiple Link hiearchy, this
        # returns the raw model variable
        var = self.var
        while issubclass(var, VariableLink):
            try:
                var = var.var
            except AttributeError:
                raise AttributeError("Tried to find raw model variables "+\
                                     "but stumbled upon a variableless link")
        return var


    def sanity_check(self, otherlink):
        # this should raise an error if the test does not pass
        return


class GraphLink(VariableLink):
    """
        This linker annotates variables with a name and
        associate them with a graph. Model is based on
        this to be customizable and ease the tossing
        around of variables.
    """
    def __init__(self, graph, var_name_dict={}):
        self.graph = graph
        self.var_name_dict = var_name_dict

    @property
    def var(self):
        return set(self.var_name_dict.keys())

    def update(self, updt):
        self.var_name_dict.update(updt)

    def get_var_wth_name(self, name):
        for key, value in self.var_name_dict.iteritems():
            if value == name:
                return key
        raise KeyError("Cannot find var named {} in graph {}".format(name, self.graph))

    def sanity_check(self, otherlink):
        if self.graph == otherlink.graph:
            raise ValueError("Two graphlinks cannot have the same name, "+\
                             "found a name collision with {}".format(self.graph))


class ParametersLink(VariableLink):
    """Indicates that this variable has these parameters linked to it
    """
    def __init__(self, var, parameters=[]):
        super(ParametersLink, self).__init__(var)
        self.parameters = assert_iterable_return_iterable(parameters)


class ExtensionLink(VariableLink):
    """Indicates that this variable is useful for that main loop extension
    """
    pass

class FrameGenLink(ExtensionLink):
    extension = 'framegen'
    def __init__(self, inputs=[], outputs=[]):
        self.inputs = assert_iterable_return_iterable(inputs)
        self.outputs = assert_iterable_return_iterable(outputs)

class TrackingLink(ExtensionLink):
    extension = 'logandsavestuff'
    def __init__(self, var, which_set=(None,)):
        super(TrackingLink, self).__init__(var)
        self.which_set = assert_iterable_return_iterable(which_set)

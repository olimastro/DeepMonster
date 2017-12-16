from deepmonster.nnet.utils import assert_iterable_return_iterable
from linkers import LinksHolder, VariableLink, GraphLink, TrackingLink

def auto_add_links(func, *args, **kwargs):
    """Use this decorator to auto add all links being
    created in a function definition (most likely a graph definition)
    to the holder
    """
    def autoaddlinks(*args, **kwargs):
        rval = func(*args, **kwargs)
        if rval is None:
            return
        model = args[0]
        model.store_links(rval)

    return autoaddlinks


class Model(object):
    """A Model consists of different computation graphs
    stiched together from inputs to outputs.

    Two major problems arise when trying to write a graph:
        - Model B can use parts of Model A, and you don't
        want to rewrite those parts.
        - A variable (potentially many) can be used in many
        different situations than the simple out of the
        box being backpropagated on. It can be a proxy
        quantity to be monitored withouth affecting
        the cost for example.

    This class is intended to solve these two flexibility
    problems by using VariableLinkers as defined below.
    """
    def __init__(self, config, architectures):
        self.config = config
        self.architectures = architectures
        self._linksholder = LinksHolder()


    def build_fprop_graph(self):
        """
            The logic of fprop is, of course, model
            dependant. The fprop should ideally
            store the output variables in a GraphLink
            of name 'costs' for bprop to know how to
            fetch the costs.
        """
        pass


    def build_bprop_graph(self):
        """
            The logic of bprop is framework dependant and has to be
            subclassed by a class using Theano, PyTorch, etc.
        """
        pass


    def build_model(self):
        print "Init Model"
        self.build_fprop_graph()
        self.build_bprop_graph()
        print "Model Done"


    def store_links(self, links):
        links = assert_iterable_return_iterable(links, 'list')
        for link in links:
            self._linksholder.store_link(link)


    @auto_add_links
    def track_var(self, var, which_set):
        return TrackingLink(var, which_set)


    @auto_add_links
    def link_var_to_graph(self, var, var_name, graph):
        var = assert_iterable_return_iterable(var)
        var_name = assert_iterable_return_iterable(var)
        assert len(var) == len(var_name), "Unsafe linking, lengths "+\
                "of vars to link and their corresponding string names are not the same"
        var_name_dict = dict(zip(var, var_name))

        # first check if that GraphLink was already made
        for link in self._linksholder:
            if isinstance(link, GraphLink) and link.graph == graph:
                link.update(var_name_dict)
                return
        # else make a new one
        return GraphLink(graph, var_name_dict)


    def fetch_var(self, var_name, graph=None):
        # if only the name of the variable is given, it will try
        # to fetch it among all the graph linkers
        for link in self.linksholder:
            if isinstance(link, GraphLink):
                if graph is not None and link.graph == graph:
                    return link.get_var_wth_name(var_name)
                else:
                    try:
                        return link.get_var_wth_name(var_name)
                    except KeyError:
                        continue
        err_msg = '' if graph is None else 'in graph ' + graph
        raise KeyError("Cannot fetch a var of with name {}".format(
            var_name + err_msg))


    @staticmethod
    def filter_for_links(locals_):
        """Typical usage is we are lazy, we don't want to sort out ourselves
        all the links and return them, let's then have this do it for us on the
        namespace!
        """
        return [val for val in locals_.values() if issubclass(val, VariableLink)]

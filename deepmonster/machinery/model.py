from . import Configurable
from deepmonster.utils import assert_iterable_return_iterable, flatten
from linkers import (LinksHelper, LinksHolder, Linker,
                     GraphLink, TrackingLink, StreamLink)

### MODEL DECORATOR ###

def auto_add_links(func, *args, **kwargs):
    """Use this decorator to auto add all links being
    created in a function definition (most likely a graph definition)
    to the holder.

    ***The decorated function needs to return Linker instances
    """
    def autoaddlinks(*args, **kwargs):
        rval = func(*args, **kwargs)
        if rval is None:
            return
        model = args[0]
        model.store_links(rval)

    return autoaddlinks


def graph_defining_method(var_names, graph='', auto_add_links=True):
    """Model decorator around a method defining a graph. By giving
    it a var_names list, it will fetch, as written in the code,
    the name of the variables and associate them in the GraphLink

    ***It needs locals() to be the returned value!
    """
    def real_decorator(func, *args, **kwargs):
        def link_up_outputs(*args, **kwargs):
            rval = func(*args, **kwargs)
            assert isinstance(rval, dict), "Return locals() if using graph_defining_method "+\
                    "decorator! (Actually, any dict will silence this error)"

            # first pick up the model variables for the graphlink
            var_name_dict = {v_val: v_name for (v_name, v_val) in \
                             rval.iteritems() if v_name in var_names}
            graphlink = GraphLink(graph, var_name_dict)

            # second, this method might also defined other links,
            # act be default like auto_add_links
            otherlinks = LinksHelper.filter_linkers(rval.values(), Linker)
            otherlinks.append(graphlink)

            model = args[0]
            model.store_links(otherlinks)

        return link_up_outputs
    return real_decorator

###-----------------###


class Model(Configurable):
    """A Model consists of different computation graphs
    stiched together from inputs to outputs.

    Two major problems arise when trying to write a graph:
        - Model B can use parts of Model A, and you don't
        want to rewrite those parts.
        - A variable (potentially many) can be used in many
        different situations than the simple out of the
        box being backpropagated on. It can be a proxy
        quantity to be monitored without affecting
        the cost for example.

    This class is intended to solve these two flexibility
    problems by using Linkers and bookeeping decorators.
    """
    def __init__(self, architecture, config):
        super(Model, self).__init__(config)
        self.architecture = architecture
        self.linksholder = LinksHolder()

    @property
    def model_parameters(self):
        return flatten([arch.params for arch in self.architecture])


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
            self.linksholder.store_link(link)


    @auto_add_links
    def add_stream(self, stream, fuel_stream):
        return StreamLink(fuel_stream, name=stream)


    @auto_add_links
    def track_var(self, var, which_set='all'):
        return TrackingLink(var, which_set)


    @auto_add_links
    def link_var_to_graph(self, var, var_name, graph):
        var = assert_iterable_return_iterable(var)
        var_name = assert_iterable_return_iterable(var)
        assert len(var) == len(var_name), "Unsafe linking, lengths "+\
                "of vars to link and their corresponding string names are not the same"
        var_name_dict = dict(zip(var, var_name))

        # first check if that GraphLink was already made
        for link in self.linksholder:
            if isinstance(link, GraphLink) and link.graph == graph:
                link.update(var_name_dict)
                return
        # else make a new one
        return GraphLink(graph, var_name_dict)


    def link_here(self, graph):
        for link in self.linksholder:
            if isinstance(link, GraphLink) and link.graph == graph:
                return link.var_name_dict


    ### ---------------------------------------------------- ###
    # these methods are, in order of complexity, methods to ease the lazy coder
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


    def fetch_vars(self, vars_name, graph=None):
        vars_name = assert_iterable_return_iterable(vars_name, 'list')
        """Fetch multiple vars and return them in the
        _same_ order of vars_names. A typical usage would then
        be at the beginning of a graph defining method:
            x, y, z = self.fetch_vars(['x','y','z'])
            ...
        """
        varz = [self.fetch_var(v, graph) for v in vars_name]
        return tuple(varz)


    def fetch_and_assign_by_exec(self, *args, **kwargs):
        """Method to almost do the ultimate lazy usage we hope to achieve.
        Writing x, y, z = self.fetch_vars(['x','y','z']) is tedious, why do
        we have to repeat the variables names we want to assign as string?

        We almost resolve this by doing in the graph defining method:
            exec self.fetch_and_assign_by_exec('x', 'y', 'z')

        This will bind the right variables in the local scope of the graph
        defining method with x, y and z.
        """
        graph = kwargs.pop('graph', None)
        assert len(kwargs) == 0, "Only graph is a valid keyword in fetch_and_assign_by_exec"
        assert all([isinstance(a, str) for a in args]), "All the variables names are \
                required as strings as argumetns"
        assign_source = str(list(args))
        assign_target = "(" + ",".join(args) + ")"

        code_string = "{} = self.fetch_vars({}, graph={})".format(
            assign_target, assign_source, graph)
        return code_string
    ### ---------------------------------------------------- ###

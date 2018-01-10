import inspect

from blocks.extensions import SimpleExtension, FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from deepmonster.blocks.extensions import *
from deepmonster.utils import assert_iterable_return_iterable, make_dict_cls_name_obj

from linkers import LinksHelper

# find all the imported extensions
EXTENSIONS = make_dict_cls_name_obj(locals().values(), SimpleExtension)

EXTENSIONS_BASE = list(SimpleExtension.BOOLEAN_TRIGGERS) + \
        list(SimpleExtension.INTEGER_TRIGGERS)

# Aside from the linkers which are specific to extensions,
# this could be any Factory for any object that get initialized with
# a config file.
class ExtensionsFactory(object):
    """This class initializes an extension with its:
        ext: string name
        configs: dict that should contain its arguments and kwargs
        linkers: if any, ExtensionLinkers that provides model variables required
        by the extension to be initialized

    By default, arguments that have no further specifications will be taken straight
    as is from the config file. There are two tricky cases to consider:
        1) Arguments that expect model variables (therefore unspecified in the config)
        2) Arguments that expect model variables _depending_ on a certain condition.
    """
    def __new__(cls, ext, config=[], linkers=[]):
        Extension = EXTENSIONS[ext]

        if len(config) == 0 and len(linkers) == 0:
            return object.__new__(Extension)

        required_args, possible_kwargs = get_extension_args_kwargs(Extension)
        if len(linkers) == 0:
            args, kwargs = set_from_config(config, required_args, possible_kwargs, True)
        else:
            args, kwargs = set_from_config_and_linkers(
                ext, config, linkers, required_args, possible_kwargs)
        return object.__new__(Extension, *args, **kwargs)

    @classmethod
    def filter_configs_linkers(cls, ext, config, linkers):
        """Help filtering the config dict and the linkers list for
        only what is usefull for this extension
        """
        config = config[ext]
        try:
            helper_cls = ExtFactoryHelper.find_helper_class(ext)
            linkers = filter(lambda x: x.__class__ in helper_cls.linkers,
                             linkers)
        except TypeError:
            # no linkers are associated with this extension
            linkers = []
        return config, linkers


    @staticmethod
    def set_from_config_and_linkers(ext, config, linkers, required_args, possible_kwargs):
        extfacthelper = ExtFactoryHelper(ext)
        # fetch everything defined in the config
        cargs, ckwargs = set_from_config(config, required_args, possible_kwargs, False)
        # fetch what the linkers have to provide
        # NOTE: For now, only loop on the required_args, technically, possible_kwargs,
        # could contain linker stuff but it is not the case now.
        largs = [] ; lkwargs = {}
        for ra in required_args:
            rval = extfacthelper.action_on(ra, config, linkers)
            if rval is None:
                continue
            largs.extend(rval[0])
            lkwargs.update(rval[1])
        if not any([k in EXTENSIONS_BASE for k in ckwargs.keys() + lkwargs.keys()]):
            lkwargs.update(extfacthelper.default_trigger)

        # now reconstruct the final args list and kwargs dict, the ones from linkers
        # will have priority
        args = []
        c_i = 0 ; l_i = 0
        for ra in required_args:
            if ra in extfacthelper.processed_arguments:
                args.append(largs[l_i])
                l_i += 1
            else:
                args.append(cargs[c_i])
                c_i += 1

        kwargs = lkwargs.update({
            k:v for k,v in ckwargs.iteritems() if not lkwargs.has_key(k)})

        return args, kwargs


    @staticmethod
    def set_from_config(config, required_args, possible_kwargs, raise_on_err):
        if raise_on_err and any([arg not in config.keys() for arg in required_args]):
            raise TypeError, "args missing to initialize extension"

        args = [config[arg] for arg in required_args]
        kwargs = {kwarg: config[kwarg] for kwarg in possible_kwargs if config.has_key(kwarg)}
        if raise_on_err and len(set(config.keys()) - (set(args) + set(kwargs.keys()))) != 0:
            raise NameError, "invalid name for a kwarg while trying to initialize extension"

        return args, kwargs


    @staticmethod
    def get_extension_args_kwargs(cls_to_init):
        """
            Return all the possible config options one could need to initialize
            the class
        """
        extension_args = []
        extension_kwargs = []
        for i, cls in enumerate(cls_to_init.__mro__):
            # do not go deeper than SimpleExtension
            if cls is SimpleExtension:
                break
            argspec = inspect.getargspec(cls.__init__)
            # seperate args and kwargs
            # We only need to retreive the args list of the first class
            # in the __mro__ since it is that one we are going to call
            # its constructor anyway. We need the kwargs of all upper
            # ones though.
            if len(argspec.defaults) > 0:
                kwargs = argspec.args[-len(argspec.defaults):]
                if i == 0:
                    # remove self
                    args = argspec.args[1:-len(argspec.defaults)]
                    extension_args.extend(args)
            else:
                kwargs = []
                if i == 0:
                    # remove self
                    args = argspec.args[1:]
                    extension_args.extend(args)

            extension_kwargs.extend(kwargs)

        extension_kwargs.extend(EXTENSIONS_BASE)

        return extension_args, extension_kwargs



class ExtFactoryHelper(object):
    """Helper class to encapsluate all the particularities out
    of the default behaviors arising when building extensions.

    These goes from default values that are not set in the
    config to values that are set in the config which are
    actuallly values that are condition on linkers.
    """
    def __new__(cls, ext):
        return object.__new__(cls.find_helper_class(ext), ext)

    @classmethod
    def find_helper_class(cls, ext):
        # get all the ExtFactoryHelper subclass defined here
        children = make_dict_cls_name_obj(globals().values(), cls)
        # find the one that defines the extension
        obj_class = None
        for child in children:
            if ext in child.extensions:
                obj_class = child
                break
        if obj_class is None:
            raise TypeError("No ExtFactoryHelper found for {}".format(ext))
        return obj_class

    def __init__(self, ext, required_arguments=[], optional_arguments=[]):
        self.serving_extension = ext
        self.required_arguments = assert_iterable_return_iterable(
            required_arguments, 'list')
        self.optional_arguments = assert_iterable_return_iterable(
            optional_arguments, 'list')
        self.processed_arguments = []

    @property
    def default_trigger(self):
        return {}

    arg_action_map = {}
    @classmethod
    def define_action_on(cls, extension_args):
        """Any method defining an action to do on some extension arguments
        have to be decorated with this decorator.
        """
        extension_args = assert_iterable_return_iterable(extension_args)
        def action_on_decorator(func):
            cls.arg_action_map.update({func: extension_args})
            return func
        return action_on_decorator


    def action_on(self, extension_arg, config, linkers):
        """Any arguments required or optional that the extension
        needs, implement this method:
            - ext_args: extension argument that needs a value
            - config: config file
            - linkers: linkers associated to this extension

        Example for DataStreamMonitoring.__init__(variables, stream):
            DSM requires variables and a stream. In the config,
            Extension['DataStreamMonitoring']['set'] is sufficient
            to have of a value of, lets say 'valid', for this whole pipeline
            to fetch the two values for variables and streams.

        returns args, kwargs for new extension
        """
        for method, args_defined_for in self.arg_action_map.iteritems():
            if extension_arg in args_defined_for and \
               not extension_arg in self.processed_arguments:
                self.processed_arguments.extend(args_defined_for)
                return method(self, config, linkers)


    def get_args_from_linkers(self, linkers, link_type, arg, filter_check=None):
        filtered_linkers = LinksHelper.filter_linkers(linkers, link_type)
        if filter_check is not None:
            filter_check(filtered_linkers)
        return LinksHelper.broadcast_request(filter_linkers, arg)


    def fetch_optional_argument(self, config, arg):
        if config.has_key(arg):
            return config[arg]

    # this should not be here
    @staticmethod
    def unique_filter_check(x):
        if len(x) != 1:
            raise ValueError

class DefaultTrigger(ExtFactoryHelper):
    @property
    def default_trigger(self):
        return {'after_epoch': True}


class DSMonitoringHelper(DefaultTrigger):
    extensions = ['DataStreamMonitoring']
    linkers = ['TrackingLink', 'StreamLink']
    def __init__(self, ext):
        super(DSMonitoringHelper, self).__init__(ext, 'set')

    @ExtFactoryHelper.define_action_on(['variables','stream'])
    def action_on_set(self, config, linkers):
        arg = config['set']
        variables = self.get_args_from_linkers(linkers, 'TrackingLink', arg)
        stream = self.get_args_from_linkers(linkers, 'StreamLink', arg)

        args = (variables, stream)
        kwargs = {'prefix': arg}
        return args, kwargs


class TDMonitoringHelper(DefaultTrigger):
    extensions = ['TrainingDataMonitoring']
    linkers = ['TrackingLink',]

    @ExtFactoryHelper.define_action_on(['variables'])
    def action_for_variables(self, config, linkers):
        arg = 'train'
        variables = self.get_args_from_linkers(linkers, 'TrackingLink', arg)

        args = (variables,)
        kwargs = {'prefix': arg}
        return args, kwargs


class FunctionHelper(DefaultTrigger):
    extensions = ['Sample', 'Reconstruction',
                  'FancyReconstruct', 'FrameGen']
    linkers = ['FunctionLink','StreamLink']
    def __init__(self, ext):
        req_arg = 'set' if ext != 'Sample' else []
        opt_arg = ['function', 'inference_set']
        super(FunctionHelper, self).__init__(ext, req_arg, opt_arg)


    # TODO: ugh, this is bad
    def fetch_optional_argument(self, config, arg):
        rval = super(FunctionHelper, self).fetch_optional_argument(config, arg)
        if rval is not None:
            filter_check = ExtFactoryHelper.unique_filter_check
        else:
            filter_check = None
        return rval, ExtFactoryHelper.unique_filter_check


    @ExtFactoryHelper.define_action_on(['func','datastream'])
    def action_on_function(self, config, linkers):
        func, filter_check = self.fetch_optional_argument(config, 'function')

        rval = self.get_args_from_linkers(
            linkers, 'FunctionLink', func, filter_check)
        inputs, outputs = rval[0]

        if self.serving_extension == 'Sample':
            func = TheanoSampling(inputs, outputs)
            return func, {}

        arg = config['set']
        stream = self.get_args_from_linkers(linkers, 'StreamLink', arg)
        # TODO if there is ever a case where we want to compute bn not on train
        bn_stream = self.get_args_from_linkers(linkers, 'StreamLink', 'train')
        func = TheanoSampling(inputs, outputs, bn_stream)

        return (func, stream), {}


class ParametersHelper(ExtFactoryHelper):
    extensions = ['SaveExperiment', 'LoadExperiment']
    linkers = ['ModelParametersLink']
    def __init__(self, ext):
        opt_arg = 'parameters_set'
        super(ParametersHeper, self).__init__(
            ext, required_arguments=[], optional_arguments=opt_arg)


    # TODO: ugh, this is bad
    def fetch_optional_argument(self, config, arg):
        rval = super(FunctionHelper, self).fetch_optional_argument(config, arg)
        if rval is not None:
            filter_check = ExtFactoryHelper.unique_filter_check
        else:
            filter_check = None
        return rval, ExtFactoryHelper.unique_filter_check


    @ExtFactoryHelper.define_action_on(['parameters'])
    def action_on_params(self, config, linkers):
        parameters_set, filter_check = self.fetch_optional_argument(config, 'parameters_set')

        params = self.get_args_from_linkers(
            linkers, 'ModelParametersLink', parameters_set, filter_check)

        return params, {}


class TheanoSampling(object):
    import theano
    from deepmonster.nnet.frameworks.popstats import get_inference_graph
    def __init__(self, theano_in, theano_out, stream=None):
        self.stream = stream
        self.theano_in = theano_in
        self.theano_out = theano_out
        self.batch_size = batch_size

    def get_stream_iterator(self):
        return self.stream.get_epoch_iterator(as_dict=True)

    def __call__(self, batch):
        # TODO:stream could be none
        print "Creating inference graph, compiling  and sampling..."
        batch_size = batch[0].shape[1]
        ifg = get_inference_graph(self.theano_in,
                                  self.theano_out,
                                  self.get_stream_iterator())
        func = theano.function(self.theano_in, ifg)
        return func(batch[0])


if __name__ == "__main__":
    get_configurable(LogAndSaveStuff)

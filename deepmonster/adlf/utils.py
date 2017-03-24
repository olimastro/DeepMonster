import numpy as np
import inspect, os, re, shutil, sys

import theano
import theano.tensor as T

rng_np = np.random.RandomState(4321)

def getftensor5():
    return T.TensorType('float32', (False,)*5)


def infer_odim_conv(i, k, s):
    return (i-k) // s + 1


def infer_odim_convtrans(i, k, s) :
    return s*(i-1) + k


def log_sum_exp(x, axis=1):
    m = T.max(x, axis=axis)
    return m+T.log(T.sum(T.exp(x-m.dimshuffle(0,'x')), axis=axis))


# http://kbyanc.blogspot.ca/2007/07/python-aggregating-function-arguments.html
def arguments(args_to_pop=None) :
    """Returns tuple containing dictionary of calling function's
    named arguments and a list of calling function's unnamed
    positional arguments.
    """
    posname, kwname, args = inspect.getargvalues(inspect.stack()[1][0])[-3:]
    posargs = args.pop(posname, [])
    args.update(args.pop(kwname, []))
    if args_to_pop is not None :
        for arg in args_to_pop :
            args.pop(arg)
    return args, posargs


# useless with new extensions
def parse_experiments(exp_root_path, save_old=False, enforce_new_name=None, prepare_files=True) :
    script_name = sys.argv[0].split('.py')[0]
    name = script_name if enforce_new_name is None else enforce_new_name

    exp_full_path = exp_root_path+name+'/'
    if prepare_files :
        print "Experiments files at " + exp_full_path

        if save_old:
            print "Save old argument passed, will move any files in this directory to a new folder"
            dirs = [x for x in os.listdir(exp_full_path) if os.path.isdir(exp_full_path + x)]
            if len(dirs) == 0:
                archive = 'run1'
            else:
                dirs = sort_by_numbers_in_file_name(dirs)
                archive = 'run' + str(int(dirs[-1].split('run')[-1]) + 1) + '/'
            cmd = 'mkdir ' + exp_full_path + archive
            print "Doing: " + cmd
            os.system(cmd)

            print "Moving all the files..."
            for f in os.listdir(exp_full_path):
                if os.path.isfile(exp_full_path + f):
                    shutil.move(exp_full_path + f,exp_full_path + archive + f)

        cmd = 'mkdir --parents ' + exp_full_path
        print "Doing: " + cmd
        os.system(cmd)
        print "Copying " + sys.argv[0] + " to " + exp_full_path+name+'.py'
        cmd = "cp " + sys.argv[0] + ' ' + exp_full_path+name+'.py'
        os.system(cmd)

    return exp_full_path, name


def prefix_vars_list(varlist, prefix):
    prefix = prefix + '_'
    return [prefix + var.name for var in varlist]


def sort_by_numbers_in_file_name(list_of_file_names):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('(\-?[0-9]+)', s) ]

    def sort_nicely(l):
        """ Sort the given list in the way that humans expect.
        """
        l.sort(key=alphanum_key)
        return l

    return sort_nicely(list_of_file_names)


def parse_tuple(tup, length=1) :
    if isinstance(tup, tuple):
        return tup
    return (tup,) * length


def get_gradients_list(feedforwards, y):
    """
        Helper function to get a list of all the gradients of y with respect to
        the output of each layer.

        Use in conjonction with fprop_passes of Feedforward while doing fprops
    """
    grads = []
    for feedforward in feedforwards:
        for fprop_pass in feedforward.fprop_passes.keys():
            # activations_list[0] is the input of the block, discard
            activations_list = feedforward.fprop_passes[fprop_pass]
            for layer, activation in zip(feedforward.layers, activations_list[1:]):
                grad = T.grad(y, activation)
                grad.name = '{}_d{}/d{}'.format(fprop_pass, y.name, layer.prefix)
                grads += [grad]

    return grads


def find_bn_params(anobject):
    """
        Helper function to find the batch norm params. It tries to be as helpful
        as it can so an object can be:
            - a layer object
            - a list of layers
            - a feedforward object
            - a list of feedforward object
            - any combination of above
    """
    # a feedforward is characterized by its 'layers' attribute
    layers = []
    if isinstance(anobject, list):
        for x in anobject:
            if hasattr(x, 'layers'):
                layers += x.layers
            else:
                layers += x
    elif hasattr(anobject, 'layers'):
        layers += anobject.layers
    else:
        layers = [anobject]

    updt = []
    for layer in layers:
        if hasattr(layer, 'bn_updates'):
            updt.extend(layer.bn_updates)
    return updt

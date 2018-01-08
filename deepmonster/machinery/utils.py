# bunch of utility objects
import theano
from deepmonster.nnet.theano.popstats import get_inference_graph

def tuplify(tup):
    if tup is not None:
        if isinstance(tup, str):
            tup = (tup,)
        else:
            assert isinstance(tup, (list, tuple))
    return tup


class VarsToTrackHolder(object):
    def __init__(self):
        self._vars_tracked = []
        self._vars_tracked_train = []
        self._vars_tracked_valid = []
        self._vars_tracked_test = []

    def get_vars_tracked(self, which_set=None):
        which_set = tuplify(which_set)
        if which_set is None:
            return self._vars_tracked
        else:
            rval = []
            for s in which_set:
                rval += getattr(self, '_vars_tracked_'+s)
            return rval

    def get_names_of_vars_tracked(self, which_set=None):
        which_set = tuplify(which_set)
        varz = self.get_vars_tracked(which_set)
        return [v.name for v in varz]

    def track_var(self, v, which_set=None):
        which_set = tuplify(which_set)
        self._vars_tracked.append(v)
        if which_set is not None:
            for s in which_set:
                getattr(self, '_vars_tracked_'+s).append(v)


# class for interfacing with Sampling extension
class Sampling(object):
    def __init__(self, stream, theano_in, theano_out, batch_size=None):
        self.stream = stream
        self.theano_in = theano_in
        self.theano_out = theano_out
        self.batch_size = batch_size

    def get_stream_iterator(self):
        return self.stream.get_epoch_iterator(as_dict=True)

    def sampling_function(self, batch):
        print "Creating inference graph, compiling  and sampling..."
        if self.batch_size is not None and batch[0].shape[1] != self.batch_size:
            batch = (batch[0][:,45:45+self.batch_size,...],)
        ifg = get_inference_graph(self.theano_in,
                                  self.theano_out,
                                  self.get_stream_iterator())
        func = theano.function(self.theano_in, ifg)
        return func(batch[0])

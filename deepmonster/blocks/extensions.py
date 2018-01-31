import cPickle as pkl
import numpy as np
import os, socket, time

from collections import OrderedDict
from scipy.misc import imsave

from blocks.extensions import SimpleExtension

class EpochExtension(SimpleExtension):
    @property
    def epoch(self):
        return self.main_loop.status['epochs_done']


class LegacyExperiment(EpochExtension):
    """
    Old way of bookeeping. Use Experiment for new way with deepmonster.machinery
    This class is intended to do all the savings and bookeeping required
    """
    def __init__(self, name, local_path=None, network_path=None, extra_infos='',
                 crush_old=False, full_dump=False, **kwargs):
        # EXPFILES_OUT has priority if set
        local_path = os.environ['EXPFILES_OUT'] if os.environ.has_key('EXPFILES_OUT') else local_path
        exp_group = os.environ['EXPFILES_GROUP'] if os.environ.has_key('EXPFILES_GROUP') else ''
        local_path = os.path.join(local_path, exp_group)
        kwargs.setdefault('before_training', True)
        super(LegacyExperiment, self).__init__(**kwargs)

        # global full_dump param for all ext if not individually set
        self.full_dump = -1 if full_dump is False else full_dump
        assert crush_old in [False, 'local', 'network', 'all']
        if network_path is None:
            assert full_dump == -1, "Full dump was given a frequency but networkpath is None"
            if crush_old == 'all':
                crush_old = 'local'
            elif crush_old == 'network':
                crush_old = False
        self.crush_old = crush_old

        self.exp_name = name
        self.local_path = os.path.join(local_path, name) + '/'
        self.network_path = network_path if network_path is None else \
                os.path.join(network_path, name) + '/'
        self.host = socket.gethostname()
        self.extra_infos = extra_infos


    def _manage_files(self, path, crush_old_cond):
        try:
            os.makedirs(path)
        except OSError:
            pass
        if len(os.listdir(path)) > 0:
            print "Files already in", path
            if self.crush_old in crush_old_cond:
                print "WARNING: Will remove them in 5s (crush_old={})".format(self.crush_old)
                time.sleep(6) # give time to the user to react
                cmd = 'rm -r {}*'.format(path)
                print "Doing:", cmd
                os.system(cmd)


    def do(self, which_callback, *args):
        if which_callback == 'before_training':
            print "Setting up experiment files"
            for ext in self.main_loop.extensions:
                if isinstance(ext, LoadExperiment) and self.crush_old != False:
                    print "WARNING: The LoadExperiment extension is in the MainLoop and the " +\
                            "flag to remove files is activated, deactivating it"
                    self.crush_old = False

            lt = time.localtime()
            timefootprint = str(lt.tm_year) + str(lt.tm_mon) + str(lt.tm_mday) + \
                    str(lt.tm_hour) + str(lt.tm_min)

            self._manage_files(self.local_path, ['local', 'all'])
            if self.network_path is not None:
                self._manage_files(self.network_path, ['network', 'all'])

            if self.network_path is not None:
                f = open(self.network_path + '{}.txt'.format(self.exp_name), 'w')
                f.write('This experiment named {} has local files on {}\n'.format(self.exp_name, self.host))
                f.write(timefootprint+'\n')
                f.write('\n')
                f.write(self.extra_infos)
                f.write('\n')


    def save(self, obj, name, ext, append_time=False, network_move=False):
        assert ext in ['npz', 'pkl', 'png']
        name = self.exp_name + '_' + name
        if append_time:
            name += str(self.epoch)
        name += '.' + ext
        tmp = os.path.join(self.local_path, 'tmp' + name)
        target = os.path.join(self.local_path, name)

        if ext == 'npz':
            np.savez(open(tmp, 'w'), obj)
        elif ext == 'pkl':
            pkl.dump(obj, open(tmp, 'w'))
        elif ext == 'png':
            # this is more fancy stuff. obj is expected to be a np array at this point
            img = prepare_png(obj)
            imsave(tmp, img)

        cmd = 'mv {} {}'.format(tmp, target)
        print "Doing:", cmd
        os.system(cmd)

        if network_move:
            t = 0
            while t<10:
                try:
                    nettarget = os.path.join(self.network_path, name)
                    cmd = 'cp {} {}'.format(target, nettarget)
                    print "Doing:", cmd
                    os.system(cmd)
                    return
                except IOError as err:
                    print "Error writing, will retry {} times".format(10-t)
                t += 1
                time.sleep(t)
            raise err


class Experiment(LegacyExperiment):
    """This extension in effect does nothing more than being a data structure
    for the other extensions and provide the save method from LegacyExperiment

    The new way of doing all the file structuring is in deepmonster.machinery.filemanager
    """
    def __init__(self, exp_name, local_path, network_path, full_dump, **kwargs):
        self.exp_name = exp_name
        self.local_path = local_path
        self.network_path = network_path
        self.full_dump = full_dump
        # bypass LegacyExperiment init
        super(EpochExtension, self).__init__(**kwargs)

    def do(self, *args):
        # do nothing
        pass



class FileHandlingExt(EpochExtension):
    """
        This extension is made to interact with the experiment extension.
        Any subclass of this one will dump its files to the experiment one
        for savings

        full_dump := freq at which a full dump will be made. A full dump
        should be a more extensive way of savings files and usually implies a move
        of those files to the servers.

        file_format := a particular file_format a child extension would
        like to use. Ex.: SaveExperiment won't care, but Sample will.
    """
    def __init__(self, full_dump=None, file_format='png', suffix=None, **kwargs):
        super(FileHandlingExt, self).__init__(**kwargs)
        self.file_format = file_format
        self._full_dump = full_dump
        if suffix is not None:
            self.suffix = suffix


    @property
    def suffix(self):
        return getattr(self, '_suffix', '')

    @suffix.setter
    def suffix(self, val):
        self._suffix = val


    def do(self, which_callback, *args) :
        if not hasattr(self, 'exp_obj'):
            self.exp_obj = self.main_loop.find_extension('Experiment')
            if self._full_dump is None:
                self.full_dump = self.exp_obj.full_dump
            else:
                self.full_dump = -1 if self._full_dump is False else self._full_dump

        if self.full_dump != -1 and self.epoch % self.full_dump == 0:
            self._do_full_dump()
        else:
            self._do()


    def _do(self):
        pass


    def _do_full_dump(self):
        # if full dump is not implemented, by default dodo with network_move
        self._do(network_move=True)



class SaveExperiment(FileHandlingExt):
    def __init__(self, parameters, save_optimizer=True,
                 save_log=True, original_save=False, **kwargs) :
        super(SaveExperiment, self).__init__(**kwargs)

        self.parameters = parameters
        self.save_optimizer = save_optimizer
        self.save_log = save_log

        self.original_save = original_save if original_save == False else True
        # if original_save is not False, it has to be either a list of epochs to
        # trigger an original_save, an int to do a modulo trigger or if True then
        # it will *always* trigger one. (does the same thing if it is 1)
        # fancy idea: accept a mix of list and int (possibly in a dict)
        if self.original_save:
            assert isinstance(original_save, list) or isinstance(original_save, int) or \
                    original_save == True, "original_save of SaveExperiment should be in " +\
                    "[list, int, True]"
            self.freq_org_save = original_save


    def isoriginal(self):
        if self.original_save:
            if isinstance(self.freq_org_save, list):
                if self.epoch in self.freq_org_save:
                    return True
            elif isinstance(self.freq_org_save, int):
                if self.epoch % self.freq_org_save:
                    return True
            elif self.freq_org_save == True:
                return True
        return False


    def _save_parameters(self, prefix='', network_move=False):
        print "Saving Parameters..."
        model_params = OrderedDict()
        for param in self.parameters :
            model_params.update(
                {param.name : param.get_value()})

        append_time = self.isoriginal()
        self.exp_obj.save(model_params, prefix + 'parameters', 'pkl',
                          append_time=append_time, network_move=network_move)


    def _save_optimizer(self, prefix='', network_move=False):
        print "Saving Optimizer..."
        # We have to save loop on the attributes of the step_rule
        optimizer_params = OrderedDict()
        for update_pair in self.main_loop.algorithm.updates:
            if update_pair[0].name is None:
                continue
            name = update_pair[0].name
            if 'OPT' in name:
                optimizer_params.update(
                    {name : update_pair[0].get_value()})

        self.exp_obj.save(optimizer_params, prefix + 'optimizer', 'pkl', network_move=network_move)


    def _save_log(self, network_move=False):
        self.exp_obj.save(
            getattr(self.main_loop, 'log'), 'main_loop_log', 'pkl', network_move=network_move)


    def _do(self):
        self._save_parameters()
        if self.save_optimizer:
            self._save_optimizer()
        if self.save_log:
            self._save_log()


    def _do_full_dump(self):
        print "Time for a full dump..."
        # this save implies a full dump of the training status (not only on a best one)
        # and with the main_loop infos so everything can be resumed properly
        self._save_parameters(network_move=True)
        self._save_optimizer(network_move=True)
        self._save_log(network_move=True)



class LoadExperiment(FileHandlingExt):
    def __init__(self, parameters, load_optimizer=True, full_load=True,
                 path=None, which_load='local', **kwargs) :
        # if path is set, it will fetch directly this path and not go through the exp object
        kwargs.setdefault('before_training', True)
        super(LoadExperiment, self).__init__(**kwargs)

        self.parameters = parameters
        self.load_optimizer = load_optimizer
        self.full_load = full_load
        self.path = path
        self.which_load = which_load


    def _do_full_dump(self):
        self._do()


    def _do(self) :
        if self.path is None:
            self.path = self.exp_obj.local_path if self.which_load is 'local' else self.exp_obj.network_path
            self.path += self.exp_obj.exp_name
        # this extension need to be used in case of requeue so if for first time launch, not crash
        if not os.path.isfile(self.path+'_parameters.pkl'):
            print "No file found, no loading"
            return
        if self.full_load :
            # a full load loads: main_loop.status, main_loop.log, parameters.pkl and optimizer.pkl
            if not self.load_optimizer:
                print "WARNING: You asked for a full load but load_optimizer is at False"
            print "Loading MainLoop log"
            ml_log = pkl.load(open(self.path+'_main_loop_log.pkl', 'r'))
            setattr(self.main_loop, 'log', ml_log)

        load_parameters(self.path + '_parameters.pkl', self.parameters)

        if self.load_optimizer:
            print "Loading Optimizer at", self.path+'_optimizer.pkl'
            optimizer_params = pkl.load(open(self.path+'_optimizer.pkl', 'r'))
            update_pair_list = self.main_loop.algorithm.updates
            for param in optimizer_params.keys():
                param_was_assigned = False
                for update_pair in update_pair_list:
                    if update_pair[0].name is None:
                        continue
                    name = update_pair[0].name
                    if param == name:
                        update_pair[0].set_value(optimizer_params[param])
                        param_was_assigned = True
                if not param_was_assigned:
                    print "WARNING: parameter "+attr_name+" of loaded optimizer unassigned!"



def load_parameters(path, parameters) :
    print "Loading Parameters at", path
    saved_parameters = pkl.load(open(path, 'r'))
    for sparam in saved_parameters.keys() :
        param_was_assigned = False
        for param in parameters:
            if param.name == sparam:
                if param.get_value().shape != saved_parameters[sparam].shape:
                    raise ValueError("Shape mismatch while loading parameters between "+\
                                     "{} of shape {} trying to load {} of shape {}".format(
                                         param.name, param.get_value().shape,
                                         sparam, saved_parameters[sparam].shape))
                param.set_value(saved_parameters[sparam])
                param_was_assigned = True
                break
        if not param_was_assigned :
            print "WARNING: parameter "+param+" of loaded parameters unassigned!"



class Sample(FileHandlingExt):
    def __init__(self, func, **kwargs) :
        self._suffix = 'samples'
        super(Sample, self).__init__(**kwargs)
        self.func = func


    def _do(self, network_move=False) :
        print "Sampling..."
        samples = self.func()
        self.exp_obj.save(samples, self.suffix, self.file_format, append_time=True,
                          network_move=network_move)



class Reconstruct(FileHandlingExt):
    def __init__(self, func, datastream, **kwargs) :
        self._suffix = 'reconstructions'
        super(Reconstruct, self).__init__(**kwargs)
        self.func = func
        self.datastream = datastream #fuel object
        self.src_done = False


    def _do(self, network_move=False) :
        print "Reconstructing..."

        data = next(self.datastream.get_epoch_iterator())
        x, reconstructions = self.func(data)
        self._do_save(x, reconstructions, network_move)


    # the do is getting deeper...
    def _do_save(self, x, reconstructions, network_move=False, only_one_src=False):
        if self.file_format == 'npz':
            out = np.concatenate((x[np.newaxis], reconstructions[np.newaxis]), axis=0)
            self.exp_obj.save(out, self.suffix, 'npz',
                              append_time=True, network_move=network_move)
        else:
            self.exp_obj.save(reconstructions, self.suffix, 'png', append_time=True,
                              network_move=network_move)
            if self.src_done:
                return
            elif only_one_src:
                append_time = False
                network_move = True
                self.src_done = True
            else:
                append_time = True
                network_move = network_move
            self.exp_obj.save(x, 'src_rec', 'png', append_time=append_time,
                              network_move=network_move)



class FancyReconstruct(Reconstruct):
    def __init__(self, func, datastream, nb_class, **kwargs) :
        super(FancyReconstruct, self).__init__(func, None, **kwargs)

        if not isinstance(datastream, list):
            datastream = [datastream]
        # infer batch size and assert there are targets
        dumzie = next(datastream[0].get_epoch_iterator())
        assert len(dumzie) == 2, "FancyReconstruct need targets, could not find"
        assert dumzie[0].shape[0] >= nb_class * 10,"Cannot use FancyReconstruct "+\
                "extension with smaller batch size than 10 * nb_class"
        nb_extra_data = dumzie[0].shape[0] - 100

        k = 10/len(datastream)
        self.nb_class = nb_class
        data = []

        # build up a dataset of 10 examples per class (not too memory hungry?)
        for j in range(len(datastream)):
            epitr = datastream[j].get_epoch_iterator()
            shape = next(epitr)[0].shape[-3:]
            _data = np.empty((k,nb_class,)+shape, dtype=np.float32)

            # usefull for extra batch size above 100
            if nb_extra_data > 0:
                extras = np.empty((nb_extra_data,) + shape, dtype=np.float32)
                extra_id = 0

            cl_accumulated = np.zeros(nb_class)

            for batches in epitr:
                targets = batches[1].flatten()
                for i, target in enumerate(targets):
                    if cl_accumulated[target] < j*k + k:
                        _data[cl_accumulated[target]%k,target,...] = batches[0][i]
                        cl_accumulated[target] += 1

                    elif nb_extra_data > 0 and extra_id < nb_extra_data:
                        extras[extra_id] = batches[0][i]
                        extra_id += 1

            data += [_data]

        assert cl_accumulated.sum() == nb_class * 10, "Could not find 10 indi"+\
                "vidual examples of each class"
        self.data = np.stack(data, axis=0)

        if nb_extra_data > 0:
            self.extra_data = extras


    def _do(self, network_move=False):
        print "Reconstructing..."
        data = self.data.reshape((10*self.nb_class,)+self.data.shape[-3:])
        if hasattr(self, 'extra_data'):
            data = np.concatenate([data, self.extra_data], axis=0)

        x, reconstructions = self.func(data)
        self._do_save(x[:100], reconstructions[:100], network_move, only_one_src=True)



class FrameGen(FileHandlingExt):
    def __init__(self, func, datastream, **kwargs) :
        kwargs.setdefault('before_first_epoch', True)
        self._suffix = 'samples'
        super(FrameGen, self).__init__(**kwargs)
        self.func = func
        self.datastream = datastream #fuel object


    def _do(self, network_move=False):
        print "Frame Generation..."

        epitr = self.datastream.get_epoch_iterator()
        while True:
            try:
                batch = next(epitr)
            except StopIteration:
                epitr = self.datastream.get_epoch_iterator()
            if np.random.randint(0,2):
                continue
            break

        samples = self.func(batch)

        self.exp_obj.save(samples, self.suffix, 'npz', append_time=True,
                          network_move=network_move)



class AdjustSharedVariable(EpochExtension):
    def __init__(self, shared_dict, **kwargs):
        super(AdjustSharedVariable, self).__init__(**kwargs)
        # shared_dict is a dictionnary with the following mapping :
        # {theano.shared : f(t, x)}
        # f(t, x) represent the new value of the shared in function of epoch and current val
        self.shared_dict = shared_dict


    def do(self, which_callback, *args) :
        for shared, func in self.shared_dict.iteritems() :
            current_val = shared.get_value()
            shared.set_value(func(self.epoch, current_val))


###----------------------###
### DEBUGGING EXTENSIONS ###
###----------------------###
class CheckDemNans(SimpleExtension):
    def __init__(self, list_to_check, **kwargs):
        self.list_to_check = list_to_check
        super(CheckDemNans, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        for x in self.list_to_check:
            if np.isnan(x.get_value()).sum() > 0:
                print "ERROR: NAN detected!"
                import ipdb ; ipdb.set_trace()


class Ipdb(SimpleExtension):
    def __init__(self, *args, **kwargs):
        # Use this extention to interrupt training and monitor
        # it on the fly. If action_obj is given, it will execute
        # a do method on it.
        self.action_obj = kwargs.pop('action', None)
        kwargs.setdefault('after_batch', True)
        super(Ipdb, self).__init__(*args, **kwargs)


    def do(self, which_callback, *args):
        # most likely we are using this in debugging so lets not
        # interrupt the flow for some error
        if self.action_obj is not None:
            try:
                rval = self.action_obj.do()
            except AttributeError as e:
                print "{} e was raised, ignoring it".format(e)
        import ipdb; ipdb.set_trace()
###----------------------###

# Useless extensions kept for legacy code. Now we dump the whole ml log.
class LogAndSaveStuff(FileHandlingExt):
        # this should be a list of strings of variables names to be saved
        # so we can follow their evolution over epochs
        # all the vars that we are trying to save should be numpy arrays!!
        # **nan_guard is not implemented anymore
    def __init__(self, arbitrary=[], train=[], valid=[], nan_guard=False,
                 log_after_batch_dump_after_epoch=False, **kwargs) :
        self._suffix = 'monitored'
        kwargs.setdefault('after_epoch', True)
        super(LogAndSaveStuff, self).__init__(**kwargs)

        def prefix_name(L, prefix):
            return [prefix + '_' + name for name in L]
        train = prefix_name(train, 'train')
        valid = prefix_name(valid, 'valid')

        self.stuff_to_save = arbitrary + train + valid
        self.didnt_log_yet = True
        self.log_after_batch_dump_after_epoch = log_after_batch_dump_after_epoch

    @property
    def iteration(self):
        return self.main_loop.status['iterations_done']


    def _do(self, network_move=False) :
        if self.didnt_log_yet:
            # assuming all iterations are created equal
            self.iterations_per_epoch = self.iteration

            dictofstuff = OrderedDict()
            for stuff in self.stuff_to_save :
                dictofstuff.update(
                    {stuff : self.fetch_stuff(stuff)})
            self.didnt_log_yet = False

        else :
            try:
                path = self.exp_obj.local_path + self.exp_obj.exp_name + '_{}.pkl'.format(self.suffix)
                f = open(path, 'r')
            except IOError:
                path = self.exp_obj.network_path + self.exp_obj.exp_name + '_{}.pkl'.format(self.suffix)
                f = open(path, 'r')
            dictofstuff = pkl.load(f)
            f.close()

            for stuff in self.stuff_to_save :
                oldnumpy_stuff = dictofstuff[stuff]
                newnumpy_stuff = np.append(oldnumpy_stuff,
                                           self.fetch_stuff(stuff))

                dictofstuff[stuff] = newnumpy_stuff

        self.exp_obj.save(dictofstuff, self.suffix, 'pkl', network_move=network_move)


    def fetch_stuff(self, stuff):
        if not self.log_after_batch_dump_after_epoch:
            return self.main_loop.log[self.iteration][stuff]

        else:
            # need to aggregate all that was recored in the
            # log for that stuff
            start = (self.epoch - 1) * self.iterations_per_epoch + 1
            end = self.epoch * self.iterations_per_epoch + 1
            stuff = [self.main_loop.log[i][stuff] for i in range(start, end, 1) if self.main_loop.log[i].has_key(stuff)]
            try:
                rval = np.concatenate(stuff)
            except ValueError:
                rval = np.stack(stuff)
            return rval



# borrowed from Kyle
def prepare_png(X):
    def color_grid_vis(X):
        ngrid = int(np.ceil(np.sqrt(len(X))))
        npxs = int(np.sqrt(X[0].size//3))
        img = np.zeros((npxs * ngrid + ngrid - 1,
                        npxs * ngrid + ngrid - 1, 3))
        for i, x in enumerate(X):
            j = i % ngrid
            i = i // ngrid
            x = tf(x)
            img[i*npxs+i:(i*npxs)+npxs+i, j*npxs+j:(j*npxs)+npxs+j] = x
        return img

    def bw_grid_vis(X):
        ngrid = int(np.ceil(np.sqrt(len(X))))
        npxs = int(np.sqrt(X[0].size))
        img = np.zeros((npxs * ngrid + ngrid - 1,
                        npxs * ngrid + ngrid - 1))
        for i, x in enumerate(X):
            j = i % ngrid
            i = i // ngrid
            x = tf(x)[:,:,0]
            img[i*npxs+i:(i*npxs)+npxs+i, j*npxs+j:(j*npxs)+npxs+j] = x
        return img

    def tf(x):
        if x.min() < -0.25:
            x = (x + 1.) / 2.
        return x.transpose(1, 2, 0)

    if X.shape[-3] == 3:
        return color_grid_vis(X)
    elif X.shape[-3] == 1:
        return bw_grid_vis(X)
    else:
        raise ValueError("What the hell is this channel shape?")

from core import Core
import os, time

class FileManager(Core):
    def configure(self):
        assert self.config.has_key('exp_name'), "Experiment needs a exp_name field"
        # there can be [exp_path or local_path] and optionaly network_path
        assert any(map(lambda x: self.config.has_key(x), ['exp_path', 'local_path'])), \
                "Experiment needs a path for its files"
        if self.config.has_key('exp_path'):
            local_path = self.config['exp_path']
        else:
            local_path = self.config['local_path']
        network_path = self.config.get('network_path', None)
        exp_name = self.config['exp_name']

        self.exp_name = exp_name
        self.local_path = os.path.join(local_path, exp_name) + '/'
        self.network_path = network_path if network_path is None else \
                os.path.join(network_path, exp_name) + '/'

        crush_old = self.config.get('crush_old', False)
        # global full_dump param for all ext if not individually set
        self.full_dump = self.config.get('full_dump', -1)
        assert crush_old in [False, 'local', 'network', 'all']
        if network_path is None:
            assert self.full_dump == -1, "Full dump was given a frequency but networkpath is None"
            if crush_old == 'all':
                crush_old = 'local'
            elif crush_old == 'network':
                crush_old = False
        self.crush_old = crush_old

        self.extra_infos = self.config.get('extra_infos', '')
        self.set_up()


    def set_up(self):
        load_flag = os.environ.get('DM_RELOAD', None)
        if load_flag is not None:
            print "WARNING: The DM_RELOAD flag has been activated but the " +\
                    "flag (crush_old) to remove files is also on, deactivating crush_old."
            self.crush_old = False

        self.manage_files(self.local_path, ['local', 'all'])
        if self.network_path is not None:
            self.manage_files(self.network_path, ['network', 'all'])

        if self.network_path is not None:
            with open(self.network_path + '{}.txt'.format(self.exp_name), 'w') as f:
                lt = time.localtime()
                timefootprint = str(lt.tm_year) + str(lt.tm_mon) + str(lt.tm_mday) + \
                        str(lt.tm_hour) + str(lt.tm_min)

                f.write('This experiment named {} has local files on {}\n'.format(self.exp_name, self.host))
                f.write(timefootprint+'\n')
                f.write('\n')
                f.write(self.extra_infos)
                f.write('\n')


    def manage_files(self, path, crush_old_cond):
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

from deepmonster.machinery.core import Core
from deepmonster.machinery.hatch import Assemble
from deepmonster.machinery.runner import TrainModel
from deepmonster.machinery.filemanager import FileManager

class DummyModel(Core):
    pass

class DummyDataFetcher(Core):
    pass


class TestAssemble(Assemble):
    @property
    def core_pieces_init_dict(self):
        cpid = {
            'model': DummyModel,
            'datafetcher': DummyDataFetcher,
            'filemanager': FileManager,
            'runner': TrainModel}
        return cpid
    @property
    def core_config_mapping(self):
        return {}

config = {
    'filemanager':
        {'exp_name': 'dm_test',
         'exp_path': '/data/dm_test/'},
    'runner':
        {'adict': {'akey': 'aval'}},
    'config':
        {'batch_size': 50},
}

assemble = TestAssemble()
ci = assemble.assemble_cores(masterdict=config, bypass_format=True)
ci['runner'].build_extensions_list()
import ipdb; ipdb.set_trace()

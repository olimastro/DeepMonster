import sys
from deepmonster.machinery.hatch import Assemble
from deepmonster.machinery.yml import YmlParser

ymlpath = sys.argv[1]
assemble = Assemble()
loader = YmlParser(ymlpath, assemble.cores_mandatory_to_init + ['config'])
loader.load()
core_instance = assemble.assemble_cores(loader)
core_instance['model'].build_model()
core_instance['runner'].run()

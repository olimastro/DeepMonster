from deepmonster.machinery.hatch import hatch, parse_config, Assemble
from deepmonster.machinery.yml import YmlLoader, YmlFieldNotFoundError
from deepmonster.utils import merge_dicts

assemble = Assemble()

core_classes = {s: None for s in assemble.core_pieces}

loader = YmlLoader('masteryml.yml', assemble.cores_mandatory_to_init)
loader.load()

# load config
config = parse_config(loader.yml['config'])

import ipdb; ipdb.set_trace()

# load mandatory classes
for req_core in assemble.cores_mandatory_to_init:
    format_parser = assemble.core_to_format_dict[req_core]
    core_classes[req_core] = loader.load_ymlentry(req_core, format_parser)

# load optional Cores
# options have 4 behaviors from the loader:
    # they return a User defined Class
    # they raise YmlFieldNotFoundError, in case we need to default
    # they return None, in case the format was good AND we need to default
    # they raise FormatError, in case an error was encountered while parsing
for opt_core in assemble.cores_with_default:
    format_parser = assemble.core_to_format_dict[opt_core]
    try:
        rval = loader.load_ymlentry(opt_core, format_parser)
        #import ipdb; ipdb.set_trace()
    except YmlFieldNotFoundError:
        rval = None
    if rval is None:
        rval = assemble.core_pieces_init_dict[opt_core]

    core_classes[opt_core] = rval

# init classes
core_instances = {s: None for s in assemble.core_pieces}
for cls in assemble.core_pieces:
    if len(assemble.core_to_format_dict[cls].particular_extra_config) > 0:
        particular_config = merge_dicts(
            config, assemble.core_to_format_dict[cls].particular_extra_config)
    else:
        particular_config = config

    core_instances[cls] = core_classes[cls](particular_config)

for instance in core_instances.values():
    for cd in instance.__core_dependancies__:
        instance.bind_a_core(cd, core_instances[cd])
import ipdb; ipdb.set_trace()

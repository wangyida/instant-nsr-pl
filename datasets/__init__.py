from termcolor import colored
import sys

datasets = {}

def register(name):

    def decorator(cls):
        datasets[name] = cls
        return cls

    return decorator


def make(name, config):
    if name == "hmvs":
        try:
            try:
                import R3DParser
            except:
                import R3DUtil
        except:
            print(
                colored(
                    "'R3DParser' or 'R3DUtils' should be explicitly included in case you are deploying on Starmap",
                    'yellow'))
            print("solution: export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/yidaw/Documents/buildboat/R3DParser/3rd/R3DLib/bin/")
        from . import hmvs
            # sys.exit()
    dataset = datasets[name](config)
    return dataset

from . import blender, colmap, dtu

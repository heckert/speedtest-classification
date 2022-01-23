import os
import numpy as np
import pathlib

def get_filenames_in_dir(dir:pathlib.Path)->list:
    '''
    Returns a list with all files but not directories in `dir`
    '''
    names = []
    for (_, _, filenames) in os.walk(dir):
        names.extend(filenames)
        break

    return names

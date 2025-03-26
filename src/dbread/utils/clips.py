import numpy as np


def clip(array, clip_num=0, inplace=False):
    if clip_num != 0:
        new_array = np.copy(array) if not inplace else array
        new_array[..., -clip_num:] = 0
        return new_array
    return array

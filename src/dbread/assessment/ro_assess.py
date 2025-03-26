from dbread.sys.rosys import rosys
from typing import Union
from numpy import ndarray
from dbread.utils.clips import clip
from dbread.utils.functions import innerprod
import numpy as np


class efficacy:
    def __init__(self, putative: rosys):
        self.putative = putative

    def run(self):
        putative = self.putative
        num_probes = self.putative.num_probes

        accuracy = []
        alignment = []
        H_coeffs = np.eye(num_probes)
        for clip_num in range(0, num_probes):
            putative.set_H(innerprod, clip(
                H_coeffs, clip_num=clip_num)).measure(plot=False)
            putative.behave(plot=False)
            assessment = putative.train_readout().test_readout()
            accuracy.append(assessment[0].statistic)
            alignment.append(assessment[1])

        return {'accuracy': accuracy, 'alignment': alignment}

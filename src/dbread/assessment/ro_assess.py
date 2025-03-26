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
        for clip_num in range(0, num_probes+1):
            H_clipped = clip(
                H_coeffs, clip_num=clip_num)
            putative.set_H(innerprod, H_clipped).measure(plot=False)
            print(f"{(H_clipped>0).sum()}")
            putative.behave(plot=False)
            assessment = putative.train_readout().test_readout()
            accuracy.append(assessment[0].statistic)
            alignment.append(assessment[1])

        return {'accuracy': accuracy, 'alignment': alignment}

import os, sys

dirScript = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirScript)
sys.path.append(os.path.join(dirScript, './../input_data_read'))

from sklearn.model_selection import train_test_split, KFold
import numpy as np
import math
import input_data_read


class ActivationFuncs:
    SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-x))
    TH = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def get_derivaty(func):
        """ Get derivaty by function activation """
        base = {
            ActivationFuncs.SIGMOID: Derivatives.dSigmoid,
        }
        return base[func]


class Derivatives:

    dSigmoid = lambda x: ActivationFuncs.SIGMOID(x) * (1 - ActivationFuncs.SIGMOID(x))
    dGTanh = lambda x: 1 - ActivationFuncs.TH(x) ** 2

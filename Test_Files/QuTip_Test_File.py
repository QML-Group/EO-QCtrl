import matplotlib.pyplot as plt
import time
import numpy as np
from qutip import *
from qutip.control import *
from qutip.qip.operations import cnot
from qutip.qip.operations import cphase
from qutip.control.grape import plot_grape_control_fields, _overlap, grape_unitary_adaptive, cy_grape_unitary
from scipy.interpolate import interp1d
from qutip.ui.progressbar import TextProgressBar


CNOT = cnot()
print(CNOT.shape)

RandomUnitary = rand_unitary(4)
print(RandomUnitary.shape)
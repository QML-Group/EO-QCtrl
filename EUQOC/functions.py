import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import scipy.sparse as sp
from alive_progress import alive_bar
from scipy.stats import unitary_group
from qutip import to_super, qpt, qpt_plot_combined, Qobj
from qutip import rand_unitary


"""

This file contains basic functions used in Energy Efficient Quantum Optimal Control

"""

def tensor(a, b):
    """
    Returns tensor product between two matrices
    """

    return np.kron(a, b)

def identity(N):
    """
    Returns Identity Matrix with Dimension 'N'
    """

    return np.identity(N)

def sigmax():
    """
    Returns Pauli-x Matrix 
    """

    return np.array([[0,1],
                     [1,0]])

def sigmay():
    """
    Returns Pauli-y Matrix 
    """

    return np.array([[0, -1j],
                     [1j, 0]])

def sigmaz():
    """
    Returns Pauli-z Matrix 
    """

    return np.array([[1, 0],
                     [0, -1]])

def cnot():
    """
    Returns CNOT Unitary Gate 
    """

    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])

def Generate_Rand_Unitary(N):
    """
    Returns N-Dimenstional Random Unitary 
    """

    x = rand_unitary(N)
    y = x.full()
    nparray = np.array(y)

    return nparray

def hadamard():
    """
    Returns Hadamard Gate
    """

    return (1/np.sqrt(2)) * np.array([[1, 1],
                                      [1, -1]])

def t_gate():

    """
    Returns T-Gate
    """

    return np.array([[1, 0],
                     [0, np.exp(-1j * (np.pi/4))]])

def rx_gate(theta):

    """
    Returns X-Rotation gate
    """

    return np.array([[np.cos(theta/2), -1j * np.sin(theta/2)],
                     [-1j * np.sin(theta/2), np.cos(theta/2)]])

def yaqq_unitary_a():

    return np.array([[-0.99891178 + 0j,  -0.01683013 - 0.04349723j],
                     [-0.0406026 + 0.02294975j, 0.7722138 + 0.63364862j]])

def yaqq_unitary_b():

    return np.array([[0.54683177 + 0j, -0.52979855 - 0.64829662j],
                     [0.8231603 + 0.15291220j, 0.26287606 + 0.47950095j]])

def rz_gate(theta):

    return np.array([[np.exp(-1j * theta/2), 0],
                     [0, np.exp(1j * theta/2)]])

def overlap(A, B):
    return np.trace(A.conj().T @ B) / A.shape[0]

def Calculate_Fidelity(U_Target, U):

    """
    Calculate Fidelity Between Target Unitary and other Unitary U

    Parameters 
    ----------

    U_Target : Target Unitary Gate 

    U: Unitary Gate to Calculate Fidelity of 

    Returns 
    ----------

    F: Fidelity between U_Target and U

    """

    F = abs(np.trace(U_Target.conj().T @ U)/np.trace(U_Target.conj().T @ U_Target))**2
    
    return F

def convert_qutip_to_numpy(operator):

    data = operator.full()
    array = np.array(data)

    return array 

def convert_qutip_list_to_numpy(operator_list):

    new_list = []

    for operator in operator_list:
        new_list.append(convert_qutip_to_numpy(operator))

    return new_list
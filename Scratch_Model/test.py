from input import *
from functions import *
from alive_progress import alive_bar

"""
This is a test file to experiment with certain functions and features

-----------------

Work in Progress

"""

grape_iter = 100
eps_range = np.array([0.1, 2 * np.pi],
                     [2 * np.pi, 4 * np.pi])

def CreateDynamicEps(R,
                     eps_e_min, eps_e_max,
                     eps_f_min, eps_f_max):

    """
    Fill array with dynamic eps values with increasing GRAPE iterations

    Input 
    --------

    GRAPE_Iterations : int | Number of GRAPE iterations

    eps_min_e = int | Minimum value of energy epsilon

    eps_max_E = int | Maximum value of energy epsilon

    eps_min_f = int | Minimum value of fidelity epsilon

    eps_max_f = int | Maximum value of fidelity epsilon
    
    Returns
    --------

    """

    eps_e_range = np.linspace(eps_e_min, eps_e_max, R)
    
    eps_f_range = np.linspace(eps_f_min, eps_f_max, R)

    eps_array = np.concatenate(eps_e_range, eps_f_range, axis = 0)

    return eps_array


test_array = CreateDynamicEps(grape_iter, eps_range[0,0], 
                              eps_range[0, 1],
                              eps_range[1,0],
                              eps_range[1,1])


print(test_array)




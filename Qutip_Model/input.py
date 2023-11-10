import numpy as np
from qutip import *
from qutip.qip.operations import cnot
from qutip.qip.operations import cphase
from qutip.qip.operations import swap

""" 

This file contains input parameters for simulating optimal control problems 
using the built-in QuTip optimal control suite and functions


"""

T = 2 * np.pi # Total gate time

Iterations_GRAPE = 100 # Total number of GRAPE iterations

Timesteps = 500 # Total number of timesteps to discretize the time space

H_Static_1 = 1 * np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) # Static Drift Hamiltonian 1

H_Static_2 = 1 * np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz()) # Static Drift Hamiltonian 2 Inlcuding Interaction Term

U_target_CNOT = cnot() # CNOT Gate 

U_target_SWAP = swap() # SWAP Gate

U_target_rand = rand_unitary(4) # Generate Random Unitary 

H_Control_1 =  [tensor(sigmax(), identity(2)), # General 2 qubit Hamiltonian with X, Y, and Z interaction terms 
         tensor(sigmay(), identity(2)),
         tensor(sigmaz(), identity(2)),
         tensor(identity(2), sigmax()),
         tensor(identity(2), sigmay()),
         tensor(identity(2), sigmaz()),
         tensor(sigmax(), sigmax()) +
         tensor(sigmay(), sigmay()) +
         tensor(sigmaz(), sigmaz())] 

H_Labels_1 = [r'$u_{1x}$', r'$u_{1y}$', r'$u_{1z}$', # Labels for H_Control_1 (optional for plotting)
            r'$u_{2x}$', r'$u_{2y}$', r'$u_{2z}$',
            r'$u_{xx}$',
            r'$u_{yy}$',
            r'$u_{zz}$'
            ] 

H_Control_2 = [tensor(sigmax(), identity(2)), # Control Hamiltonian 2: no Y terms 
               tensor(sigmaz(), identity(2)),
               tensor(identity(2), sigmax()),
               tensor(identity(2), sigmaz()),
               tensor(sigmax(), sigmax()) +
               tensor(sigmaz(), sigmaz())
               ] 

H_Labels_2 = [r'$u_{1x}$', r'$u_{1z}$', # Labels for H_Control_2 (optional for plotting)
              r'$u_{2x}$', r'$u_{2z}$', 
              r'$u_{xx}$', r'$u_{zz}$'] 

H_Control_3 = [tensor(sigmax(), sigmax()), # Control Hamiltonian 3: No single qubit terms
               tensor(sigmay(), sigmay()), 
               tensor(sigmaz(), sigmaz())]

H_Labels_3 = [r'$u_{xx}$', # Labels for H_Control_3 (optional for plotting)
              r'$u_{yy}$', 
              r'$u_{zz}$'] 

H_Control_4 = [tensor(sigmax(), identity(2)), # Control Hamiltonian 2: no Y and Z terms
               tensor(identity(2), sigmax()),
               tensor(sigmax(), sigmax())
               ] 

H_Labels_4 = [r'$u_{1x}$', # Labels for H_Control_4 (optional for plotting)
              r'$u_{2x}$', 
              r'$u_{xx}$'] 

H_Control_5 = [tensor(sigmax(), identity(2)), # Control Hamiltonian 5: full control without Y-control
               tensor(sigmaz(), identity(2)),
               tensor(identity(2), sigmax()),
               tensor(identity(2), sigmaz()),
               tensor(sigmax(), sigmax()),
               tensor(sigmax(), sigmaz()),
               tensor(sigmaz(), sigmax()),
               tensor(sigmaz(), sigmaz())]

H_Labels_5 = [r'$u_{1x}$', r'$u_{1z}$',
              r'$u_{2x}$', r'$u_{2z}$',
              r'$u_{xx}$', r'$u_{xz}$',
              r'$u_{zx}$', r'$u_{zz}$'] 


print((H_Control_4[0].dag() * H_Control_4[0]).tr())
import numpy as np
from functions import *

"""

This File Contains Inputs that are used in the "main.py" file to calculate and simulate
Quantum Optimal Control problems in python

-----------------

Work in Progress

"""

T = 2 * np.pi # Total gate time

Iterations_GRAPE = 100 # Total number of GRAPE iterations

Timesteps = 500 # Total number of timesteps to discretize the time space

H_Static_1 = np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz()) # Static Drift Hamiltonian 1 Inlcuding Interaction Term

U_Target_CNOT = cnot()

U_Target_Rand = rand_unitary(4)

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

import numpy as np
from functions import * 

"""

This File Contains Inputs that are used in the "main.py" file to calculate and simulate
Quantum Optimal Control problems in python

-----------------

Work in Progress

"""

T = 2 * np.pi

Timesteps = 500

GRAPE_Iterations = 500

eps_f = 1

eps_e = 100

H_Static_Ising = 1 * np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz()) # Static Drift Hamiltonian 1 Inlcuding Interaction Term

U_Target_CNOT = cnot()

U_Target_Rand = Generate_Rand_Unitary(4)

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

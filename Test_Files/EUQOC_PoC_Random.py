import jax
from jax import numpy as jnp
import pennylane as qml
import matplotlib.pyplot as plt
from functools import partial
import time
import optax
from pennylane import utils

# Define Block Pulse Shape

def block_pulse(t, Omega, t_0, t_1):
    return Omega * jnp.heaviside(t - t_0, 1.0) * jnp.heaviside(t_1 - t, 1.0)

# Define Pulse Shape based on multiple Block Pulses

def Pulse(params, t, max_amp = 1.0):

    P = len(params) // 3
    amps, times = jnp.split(params, [P])
    block_pulses = 0
    for i in range(amps.size):
        block_pulses += block_pulse(t, amps[i], times[2*i], times[2*i+1]) 

    block_pulses_normalized = block_pulses
    return max_amp * block_pulses_normalized

# Unitary Evolution of Hamiltonian

def pulse_matrix(params):
    return qml.evolve(H, mxstep=1000)(params, T).matrix()

# Calculate Fidelity between Unitary and Target Unitary 

@jax.jit
def profit(params):

    op_mat = pulse_matrix(params)

    return jnp.abs(jnp.trace(target.conj().T @ op_mat)) / 2**num_wires

grad = jax.jit(jax.grad(profit))

# Calculate Hamiltonian Norm

def Calc_Norm(H):
    return jnp.linalg.norm(H)

# Translate Pennylane Hamiltonian to Numpy Array

def ToNumpy(H):
    return H.sparse_matrix().toarray()

# Function to Calculate Energetic Cost of a Hamiltonian

def CalcEnergeticCost(H, params, t):

    EnergeticCost = 0
    stepsize = max(t)/len(t)
    Energetic_Cost_List = []
    for i in t:
        EnergeticCost += Calc_Norm(ToNumpy(H(params, i)))*stepsize
        Energetic_Cost_List.append(EnergeticCost)
        #print(f"Cost in timestep {i} is", EnergeticCost)

    return EnergeticCost, Energetic_Cost_List

grad_energetic_cost = jax.jit(jax.grad(CalcEnergeticCost))

##################################################################################################################################

# Initial Values and Parameters 

T = 2 * jnp.pi 
timesteps = jnp.linspace(0, T, 1000)
max_amp = 1.0

amps = jnp.array([-2.0, -1.0, 2.0, 3.0])
times = jnp.array([0.2, 0.8, 1.0, 1.8, 2.0, 3.8, 4.0, 5.9])
params = jnp.hstack([amps, times])
PulseShape = Pulse(params,timesteps,max_amp=max_amp)
plt.plot(timesteps, PulseShape)
plt.xlabel('Time t')
plt.ylabel('Control Pulse Amplitude (V)')
plt.title('Input Control Pulse')
plt.show()
# Calculate Control Parameters

S_k = partial(Pulse, max_amp=max_amp)

# Define Pauli Operators in Hamiltonian

X, Y, Z = qml.PauliX, qml.PauliY, qml.PauliZ
num_wires = 2
ops_H_d = [Z(0), Z(1)]
ops_param = [Z(0), X(1), Y(1), Z(1), Z(0) @ X(1)]

# Define Hamiltonian Coefficients 

coeffs = [1.0, 1.0] + [S_k for op in ops_param]

# Define Hamiltonian with Coefficients and Pauli Operators 

ops_total = ops_H_d + ops_param
H = qml.dot(coeffs, ops_total)

# Define Tolerances 

atol = rtol = 1e-10

# Define Target Unitary 

target = qml.CNOT([0,1]).matrix()
target_name = "CNOT"
print(f"Our target unitary is a {target_name} gate, with matrix \n{target.astype('int')}")

# Calculate Fidelity Between Target Unitary and Control Unitary 

grad = jax.jit(jax.grad(profit))
params = [params] * len(ops_param)
arb_mat = jnp.round(pulse_matrix(params), 4)
arb_profit = profit(params)
print(f"Fidelity is: {arb_profit}")

# Calculate Energetic Cost of Unitary 

"""Energetic Cost Plot
Energetic_Cost, List = CalcEnergeticCost(H, params, timesteps)
print("The Energetic Cost is:", Energetic_Cost)
plt.plot(timesteps, List)
plt.title(f"Total Energetic Cost is {Energetic_Cost} J \n Gate Fidelity is {arb_profit*100} %")
plt.xlabel("Time t")
plt.ylabel("Energetic Cost (a.u.)")
plt.show()
"""
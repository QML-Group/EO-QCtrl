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

##################################################################################################################################

# Fidelity Optimization Parameters

P = 3 # Number of Block Pulses

times = [jnp.linspace(0, T, P * 2) for op in ops_param] # Initial Parameters for the start and end times of the block pulses

params = [jnp.hstack([[0.1 * (-1) ** i for i in range(P)], time]) for time in times] # Initial Parameters: small alternating amplitudes and times

def run_adam(profit_fn, grad_fn, params, learning_rate, num_steps):

    start_time = time.process_time()

    optimizer = optax.adam(learning_rate, b1=0.97)
    opt_state = optimizer.init(params)

    hist = [(params.copy(), profit_fn(params))]
    for step in range(num_steps):
        g = grad_fn(params)
        updates, opt_state = optimizer.update(g, opt_state, params)

        params = optax.apply_updates(params, updates)
        hist.append([params, c := profit_fn(params)])
        if (step + 1) % (num_steps // 10) == 0:
            print(f"Step {step+1:4d}: {c:.6f}")
    _, profit_hist = list(zip(*hist))

    plt.plot(list(range(num_steps + 1)), profit_hist)
    ax = plt.gca()
    ax.set(xlabel="Iteration", ylabel=f"Fidelity $(p)$")
    plt.show()
    end_time = time.process_time()
    print(f"The optimization took {end_time-start_time:.1f} (CPU) Seconds")

    max_params, max_profit = hist[jnp.argmax(jnp.array(profit_hist))]
    
    return hist, max_params, max_profit


learning_rate = -0.2
num_steps = 500
hist, opt_params, opt_profit = run_adam(profit, grad, params, learning_rate, num_steps )

print(opt_params, opt_profit)

""" Energetic Cost Plot
Energetic_Cost, List = CalcEnergeticCost(H, opt_params, timesteps)
print("The Energetic Cost is:", Energetic_Cost)

plt.plot(timesteps, List)
plt.title(f"Total Energetic Cost is {Energetic_Cost} J \n Gate Fidelity is {opt_profit*100} %")
plt.xlabel("Time t")
plt.ylabel("Energetic Cost (a.u.)")
plt.show()
"""

colors = {0: "#70CEFF", 1: "#C756B2", 2: "#FDC357"}
dashes = {"X": [10, 0], "Y": [2, 2, 10, 2], "Z": [6, 2]}

def plot_optimal_pulses(hist, pulse_fn, ops, T, target_name):
    _, profit_hist = list(zip(*hist))
    fig, axs = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"hspace": 0.0}, sharex=True)

    max_params, max_profit = hist[jnp.argmax(jnp.array(profit_hist))]
    plot_times = jnp.linspace(0, T, 300)
    for p, op in zip(max_params, ops):
        label = op.name
        ax = axs[0] if isinstance(label, str) else axs[1]
        if isinstance(label, str):
            label = f"${label[-1]}_{op.wires[0]}$"
            dash = dashes[label[1]]
        else:
            label = "$" + " ".join([f"{n[-1]}_{w}" for w, n in zip(op.wires, label)]) + "$"
            dash = [10, 0]
       
        col = colors[op.wires[0]]
        
        values = [pulse_fn(p, t) for t in plot_times]
        ax.plot(plot_times, values, label=label, dashes=dash, color=col)
    ax.legend()
    axs[0].legend(title="Single-qubit terms", ncol=int(jnp.sqrt(len(ops))))
    axs[1].legend(title="Two-qubit terms")
    title = f"{target_name}, Fidelity={max_profit:.6f}"
    axs[0].set(ylabel=r"Pulse function $f(p, t)$", title=title)
    axs[1].set(xlabel="Time $t$", ylabel=r"Pulse function $S_k(p, t)$")
    plt.show()


plot_optimal_pulses(hist, S_k, ops_param, T, target_name)
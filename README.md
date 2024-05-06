# EUQOC: Energy Efficient Universal Quantum Optimal Control

This repository contains the software for the Master Thesis of Sebastiaan Fauquenot: "Energy Efficient Universal Quantum Optimal Control"

### Abstract

Quantum optimal control is a rapidly growing field with diverse methods and applications. In
this work, the possibility of using quantum optimal control techniques to co-optimize the energetic
cost and the process fidelity of a quantum unitary gate is investigated. The theoretical definition and
quantization of quantum unitary gates, as well as the relationship between the process fidelity and
the energetic cost of a quantum unitary gate are explored. Two different quantum optimal control
methods to co-optimize both fidelity and energetic cost, i.e., the Gradient Ascent Pulse Engineering
method and model-free Deep Reinforcement Learning are investigated. The performance of
both quantum optimal control techniques in the presence of noise is probed. We find that the energetic
cost of a quantum unitary gate can be quantized by integrating the control pulses and norm of the
corresponding Hamiltonian operators over the total time duration of the unitary, and for single qubit
gates by calculating the arc length of the quantum unitary gate on the Bloch sphere. A Pareto
optimal front between the process fidelity and the energetic cost of a quantum gate is identified, where
a lower energetic cost yields an inherently lower process fidelity. A python package called ”EUQOC”
(Energy Efficient Universal Quantum Optimal Control) has been created to implement energy optimal
quantum gate synthesis, both with the Energy Optimal Gradient Ascent Pulse Engineering (EO-GRAPE)
method and by model-free Deep Reinforcement Learning. It is found that the EO-GRAPE method
performs better than the reinforcement learning methods, for all noise settings and neural network
sizes. For future work, the optimization problem could be translated to the frequency domain to
increase the computational efficiency. Furthermore, the relationship between information and energy
can be investigated by looking at the complexity of the pulse or the decomposition of the quantum
unitary gate.


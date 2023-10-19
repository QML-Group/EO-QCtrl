import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.interpolate import interp1d, interp2d

timesteps = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500]
EC = [187.05051764472736, 50.420159546442015, 48.87883669124049, 49.536472578131246, 43.926669077185466, 45.9275789459136, 41.07273846210889, 38.26016246337677, 36.93743885477965, 35.68583996991774, 32.70198871124866, 27.503299242170918, 27.11934424422788, 27.45246913264507]
F = [0.6392287613759335, 0.34673420453342646, 0.5668501022849716, 0.5931308440243787, 0.736599838650287, 0.7197777700250999, 0.7532992712253753, 0.8319857048506297, 0.8484303074812057, 0.8345913231158861, 0.8744753109218228, 1.0000000000000018, 1.0000000000000022, 1.0000000000000004]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of Timestep Iterations')
ax1.set_ylabel('Process Fidelity', color=color)
lns1 = ax1.plot(timesteps, F, color=color, label='Process Fidelity', linestyle = '-', marker = 'D')
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  

color = 'tab:blue'
ax2.set_ylabel('Energetic Cost (a.u.)', color=color) 
lns2 = ax2.plot(timesteps, EC, color=color, label='Energetic Cost', linestyle = '-', marker = 'D')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout() 
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='center right')
plt.title('Process Fidelity and Energetic Cost of Random Unitary versus Timestep Iterations')
plt.show()
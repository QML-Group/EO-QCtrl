from EUQOC_PoC_Optimize import *
from EUQOC_PoC_Random import *

Energetic_Cost_Optimal, List_Optimal = CalcEnergeticCost(H, opt_params, timesteps)
Energetic_Cost_Random, List_Random = CalcEnergeticCost(H, params, timesteps)
arb_profit_random = profit(params)
arb_profit_optimal = profit(opt_params)
print("Random Pulse Fidelity is:, ", arb_profit_random, "Optimal Pulse Fidelity is: ", arb_profit_optimal)

plt.plot(timesteps, List_Random, label = "Random Pulse Energetic Cost", color = "red")
plt.plot(timesteps, List_Optimal, label = "Optimal Pulse Energetic Cost", color = "green")
plt.title(f"$F_O$ = {round(arb_profit_optimal, ndigits = 1)} %, $F_R$ is {round(arb_profit_random, ndigits = 1)} %, \n $E_O$ = {round(Energetic_Cost_Optimal, ndigits = 1)} a.u., $E_R$ = {round(Energetic_Cost_Random, ndigits = 1)} a.u.")
plt.xlabel("Time t")
plt.ylabel("Energetic Cost (a.u.)")
plt.legend()
plt.grid()
plt.show()
from qutip import *

test = basis(4, 0)
density = test * test.dag()
print(density)
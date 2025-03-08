import numpy as np


n = 2			# order of root
xi = 10			# initial guess
a = 3			# root of the number a

for i in range(10):
	x0 = 0.5 * (xi + a/xi)

	print(f'iteration {i}: x0 = {x0:.15f}')
	xi = x0

print(f'\nnumpy: {np.sqrt(a):.15f}, my: {x0:.15f}')
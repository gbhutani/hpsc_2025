
n = 2			# order of root
xi = 10			# initial guess
a = 25			# root of the number a :--> a**(1/n)


def f(x):
	return x**n - a

def fp(x):
	return n * x**(n-1)

for i in range(10):
	x0 = xi - f(xi) / fp(xi)

	print(f'iteration {i}: x0 = {x0:.15f}')
	xi = x0



print(f'\ncorrect value: {a ** (1/n):.15f}, my_estimation : {x0:.15f}')
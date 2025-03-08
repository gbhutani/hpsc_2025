


def nth_order_root(n, a, xi=10, debug=False):
	"""
	Calculates the nth order root of a.
	
	n (real)    : order of root.
	a (real)    : root of which number.
	xi (real)   : initial guess value. Optional, default 10.
	debug (bool): allows printing iterative results. Optional, default False.

	returns		: nth order root of a.
	"""
	def f(x):
		return x**n - a

	def fp(x):
		return n * x**(n-1)

	for i in range(10):
		x0 = xi - f(xi) / fp(xi)

		if debug:
			print(f'iteration {i}: x0 = {x0:.15f}')
		xi = x0

	if debug: print(f'\ncorrect value: {a ** (1/n):.15f}, my_estimation : {x0:.15f}')
	return x0

print(nth_order_root(n=2, a=4, debug=True))
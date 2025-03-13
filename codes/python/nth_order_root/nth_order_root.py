from numpy import nan, power


def nth_root(n, x, init_guess=10, k_max=100, tolerance=1.0e-14, debug=False):
	"""
	Computes the nth order root of a given number 'x', using Newton's method.
	
	Parameters:
	-----------
	n : float
	    The order of the root (e.g., 2 for square root, 3 for cube root).
	x : float
	    The number whose nth root is to be calculated.
	init_guess : float, optional (default=10)
	    Initial guess for Newton's iteration.
	tolerance : float, optional (default=1.0e-14)
	    Convergence criterion for relative error.
	debug : bool, optional (default=False)
	    If True, prints iteration details.

	Returns:
	--------
	float
	    The computed nth root of 'x'.

	Notes:
	------
	- Uses Newton-Raphson iteration: "s_{i+1} = s_i - f(s_i) / f'(s_i)".
	- Stops iterating if the relative error is below the specified 'tolerance'.
	- If 'debug=True', prints the progress of each iteration.
	"""
	if x == 0:
		return 0
	elif x < 0:
		return nan

	f = lambda s: s**n - x
	fp = lambda s: n * s**(n-1)

	s_i = init_guess
	for i in range(k_max):
		s_0 = s_i - f(s_i) / fp(s_i)
    	
		rel_err = abs((s_0 - s_i) / s_i)
		if debug: print(f'iteration {i}: s_i = {s_0:.15f}, rel_err = {rel_err:.5e}')
		
		if rel_err <= tolerance: break

		s_i = s_0
	if debug: print(f'correct value: {power(x, 1/n):.15f}, my_estimation : {s_0:.15f}\n')
	return s_0

nth_root(n=3, x=1e8, debug=True)

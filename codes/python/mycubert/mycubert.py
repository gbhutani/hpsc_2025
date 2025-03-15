def cube_root(x, debug=False):
	if x==0:
		return 0
	s=1.0
	kmax=100
	tol=1.0e-14
	for k in range(kmax):
		if debug:
			print(f"At iteration {k} the value of s={s:20.15f}")
		s0 = s
		s = (2./3)*s + (1./3)*(x/s**2)
		delta_s = s-s0
		if(abs(delta_s/x) < tol):
			break
	if debug:
		print(f"Finally, the value of s={s:20.15f}")
	return s
	
if __name__=="__main__":
	import test_case
	print("In main, executing test() in test_case.py")
	test_case.test()
	

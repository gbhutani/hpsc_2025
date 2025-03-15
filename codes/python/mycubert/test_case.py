def test():
	from mycubert import cube_root
	import math
	small=1.0e-14
	xvalues=[-8.0,0,8,1000,0.001, 1e6]
	cubevalues=[-2.0,0,2,10,0.1, 1e2]
	for x,c_analytical in zip(xvalues,cubevalues):
		c=cube_root(x)
		print(f"for x={x}, c={c} and c_analytical={c_analytical}")
		assert (c-c_analytical)<small, f"cube root disagrees with analytical cube root for x={x}"
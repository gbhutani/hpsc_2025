def test_1():
    from nth_order_root import nth_root
    from numpy import power

    n = 3
    xvalues = [2, 100, 10000, 0.0001, 0, 1e7]

    small=1.0e-13
    for x in xvalues:
        root = nth_root(n, x)
        root_numpy = power(x, 1/n)
        print(f"{n=}: for x={x}, my_root={root} and numpy_root={root_numpy}")
        assert (root-root_numpy) < small, f"{n}th root disagrees with numpy root for x={x}"


    
def test_2():
    from nth_order_root import nth_root
    from numpy import power

    n = 2
    xvalues = [2, 100, 10000, -5]

    small=1.0e-13
    for x in xvalues:
        root = nth_root(n, x)
        root_numpy = power(x, 1/n)
        print(f"{n=}: for x={x}, my_root={root} and numpy_root={root_numpy}")
        assert (root-root_numpy) < small, f"{n}th root disagrees with numpy root for x={x}"


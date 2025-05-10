program evaluate_y
	
	use omp_lib			! importing something in program should go first
	implicit none
	integer :: i
	integer :: n_threads
	integer, parameter :: n = 100000
	real(kind=8), dimension(n) :: y
	real(kind=8) :: x, dx
	
	dx = 1.d0 / n
	
	!$ n_threads = THREADS
	
	!$ call omp_set_num_threads(n_threads)
	
	print *, "using OpenMP with threads =", n_threads 
	!$omp parallel do private(x)
	do i = 1, n
		x = i * dx
		y(i) = sin(x) * exp(-x**2)
	enddo
	!$omp end parallel do
	
	print *, 'y array has been filled with n=', n
	
	! compile as:  gfortran -cpp -fopenmp -DTHREADS=4 eval_y.f90
end program evaluate_y

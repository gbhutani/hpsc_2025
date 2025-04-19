program frist_openmp

	use omp_lib
	implicit none
	
	integer :: thread_number
	
	call omp_set_num_threads(5)
	!$omp parallel
	!$omp critical
	thread_number = omp_get_thread_num()
	print *, "this is thread: ", thread_number
	!$omp end critical	
	!$omp end parallel


end program

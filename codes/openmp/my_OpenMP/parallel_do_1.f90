program parallel_do_1
	
	use omp_lib
	implicit none
	
	integer :: i, n = 10
	
	call omp_set_num_threads(5)
	
	! the omp parallel do distributes the loop index over threads
	
	!$omp parallel do
	do i=1, n
		print *, i, ": do loop in ", omp_get_thread_num()
	enddo
	!$omp end parallel do
	
	
end program parallel_do_1

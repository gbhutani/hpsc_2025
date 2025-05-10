PROGRAM OpenMp_reduction_sum

	use omp_lib
	implicit none
	
	integer :: i, sum
	integer :: n_threads, thread_num
	
	!$ call omp_set_num_threads(5)
	!$omp parallel private(i, sum, thread_num)
		sum = 0
		thread_num = omp_get_thread_num()
	
		!$omp do
		do i=1, 100
			sum = sum + i
		end do
		!$omp end do
	
		!$omp critical
		print *, "thread ", thread_num, " sum = ", sum
		!$omp end critical
	!$omp end parallel

END PROGRAM OpenMp_reduction_sum

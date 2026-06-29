cutensor: benchmark.cu                                                                                                               
	nvcc benchmark.cu -o cutensor_benchmark \
	   -I/lustre/home/extusers/udle1b776/devel/tapp/jan/reference-implementation/api/include/ \
	   -L/lustre/home/extusers/udle1b776/devel/tapp/jan/reference-implementation/build_cutensor_cublas_tblis_emulated/cutensor_bindings \
           -arch=sm_100 \
	   -lcurand -lcublas -ltapp-cutensor -Xcompiler -fopenmp
cublas_eager: benchmark.cu                                                                                                               
	nvcc benchmark.cu -o cublas_eager_benchmark \
	   -I/lustre/home/extusers/udle1b776/devel/tapp/jan/reference-implementation/api/include/ \
	   -L/lustre/home/extusers/udle1b776/devel/tapp/jan/reference-implementation/build_cublas_emulated_eager/cublas_bindings \
           -arch=sm_100 \
	   -lcurand -lcublas -ltapp-cublas -Xcompiler -fopenmp
cublas: benchmark.cu                                                                                                               
	nvcc benchmark.cu -o cublas_benchmark \
	   -I/lustre/home/extusers/udle1b776/devel/tapp/jan/reference-implementation/api/include/ \
	   -L/lustre/home/extusers/udle1b776/devel/tapp/jan/reference-implementation/build_cutensor_cublas_tblis_emulated/cublas_bindings \
           -arch=sm_100 \
	   -lcurand -lcublas -ltapp-cublas -Xcompiler -fopenmp

cutensor: benchmark.cu                                                                                                               
	nvcc benchmark.cu -o cutensor_benchmark \
	   -I/lustre/work/pdl17866/shared/lee-maximee/reference-implementation/api/include/ \
	   -L/lustre/work/pdl17866/shared/lee-maximee/reference-implementation/build/cutensor_bindings \
           -arch=sm_100 \
	   -lcurand -lcublas -ltapp-cutensor -Xcompiler -fopenmp
cublas_eager: benchmark.cu                                                                                                               
	nvcc benchmark.cu -o cublas_eager_benchmark \
	   -I/lustre/work/pdl17866/shared/lee-maximee/reference-implementation/api/include/ \
	   -L/lustre/work/pdl17866/shared/lee-maximee/reference-implementation/build/cublas_bindings \
           -arch=sm_100 \
	   -lcurand -lcublas -ltapp-cublas -Xcompiler -fopenmp
cublas: benchmark.cu                                                                                                               
	nvcc benchmark.cu -o cublas_benchmark -DUSE_CUBLAS=1\
	   -I/lustre/work/pdl17866/shared/lee-maximee/reference-implementation/api/include \
	   -I/lustre/work/pdl17866/shared/lee-maximee/reference-implementation/cublas_bindings/include \
	   -L/lustre/work/pdl17866/shared/lee-maximee/reference-implementation/build/cublas_bindings \
           -arch=sm_100 \
	   -lcurand -lcublas -ltapp-cublas -Xcompiler -fopenmp
all: cublas cublas_eager cutensor

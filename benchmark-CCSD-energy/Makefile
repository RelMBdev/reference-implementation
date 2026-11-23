cutensor: benchmark.cu                                                                                                               
	nvcc benchmark.cu -o cutensor_benchmark --extended-lambda \
	   -I${TAPP_INCLUDE_PATH} \
	   -I${TAPP_API_INCLUDE_PATH} \
	   -L${TAPP_CUTENSOR_LIBRARY_PATH} \
	   -I${HDF5INCLUDE} \
	   -L${HDF5DIR} \
           -arch=sm_100 \
	   -lhdf5_cpp -lhdf5 -lcutensor -lcurand -lcublas -ltapp-cutensor -Xcompiler -fopenmp
cublas: benchmark.cu                                                                                                               
	nvcc benchmark.cu -o cublas_benchmark --extended-lambda -DUSE_CUBLAS=1 \
	   -I${TAPP_API_INCLUDE_PATH} \
	   -I${TAPP_CUBLAS_INCLUDE_PATH} \
	   -L${TAPP_CUBLAS_LIBRARY_PATH} \
	   -I${HDF5INCLUDE} \
	   -L${HDF5DIR} \
           -arch=sm_100 \
	   -lhdf5_cpp -lhdf5 -lcutensor -lcurand -lcublas -ltapp-cublas -Xcompiler -fopenmp
cublas_OS_II: benchmark.cu                                                                                                               
	nvcc benchmark.cu -o cublas_benchmark_OS_II --extended-lambda -DUSE_CUBLAS=1 \
	   -I${TAPP_API_INCLUDE_PATH} \
	   -I${TAPP_CUBLAS_INCLUDE_PATH} \
	   -L${TAPP_CUBLAS_OS_II_LIBRARY_PATH} \
	   -I${HDF5INCLUDE} \
	   -L${HDF5DIR} \
           -arch=sm_100 \
	   -lhdf5_cpp -lhdf5 -lcutensor -lcurand -lcublas -ltapp-cublas -Xcompiler -fopenmp
all: cublas cublas_eager cutensor

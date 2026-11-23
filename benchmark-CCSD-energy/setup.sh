module purge
module load gmp/6.3.0
module load mpfr/4.2.1
module load mpc/1.3.1
module load gcc/14.2.0
module load slurm/slurm/25.11
module load cuda13.1/toolkit
module load nvhpc-hpcx-cuda13/26.1 
module load hdf5/1.14.0

export TAPP_INCLUDE_PATH=/lustre/work/pdl17866/shared/lee-maximee/reference-implementation/api/include/tapp
export TAPP_API_INCLUDE_PATH=/lustre/work/pdl17866/shared/lee-maximee/reference-implementation/api/include
export TAPP_CUBLAS_INCLUDE_PATH=/lustre/work/pdl17866/shared/lee-maximee/reference-implementation/cublas_bindings/include

export TAPP_CUBLAS_LIBRARY_PATH=/lustre/work/pdl17866/shared/lee-maximee/reference-implementation/build/cublas_bindings
export TAPP_CUBLAS_OS_II_LIBRARY_PATH=/lustre/work/pdl17866/shared/lee-maximee/reference-implementation/build_OS_II/cublas_bindings
make cublas_OS_II

export TAPP_LIBRARY_PATH=/lustre/work/pdl17866/shared/lee-maximee/reference-implementation/build_OS_II/cublas_bindings
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TAPP_LIBRARY_PATH

#!/bin/bash

#SBATCH --exclusive 
#SBATCH --cpus-per-task=144
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00

module purge
module load gmp/6.3.0
module load mpfr/4.2.1
module load mpc/1.3.1
module load gcc/14.2.0
module load slurm/slurm/24.11
module load cuda13.1/toolkit
module load nvhpc-hpcx-cuda13/26.1
module load hdf5/1.14.0

export LD_LIBRARY_PATH=/lustre/work/pdl17866/shared/lee-maximee/reference-implementation/build/cublas_bindings:$LD_LIBRARY_PATH

#./cublas_benchmark 123  1 20 0 64 0 0  > PbS_precision_OS_II.out
#./cublas_benchmark 123  2 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123  3 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123  4 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123  5 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123  6 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123  7 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123  8 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123  9 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123 10 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123 11 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123 12 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123 13 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123 14 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123 15 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123 16 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123 17 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123 18 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123 19 20 0 64 0 0 >> PbS_precision_OS_II.out
#./cublas_benchmark 123 20 20 0 64 0 0 >> PbS_precision_OS_II.out

./cublas_benchmark 123  5 100 1 64 0 0  > PbS_timings.out
./cublas_benchmark 123  6 100 1 64 0 0 >> PbS_timings.out
./cublas_benchmark 123  7 100 1 64 0 0 >> PbS_timings.out
./cublas_benchmark 123  8 100 1 64 0 0 >> PbS_timings.out
./cublas_benchmark 123  9 100 1 64 0 0 >> PbS_timings.out
./cublas_benchmark 123 10 100 1 64 0 0 >> PbS_timings.out
./cublas_benchmark 123 11 100 1 64 0 0 >> PbS_timings.out
./cublas_benchmark 123 12 100 1 64 0 0 >> PbS_timings.out
./cublas_benchmark 123 13 100 1 64 0 0 >> PbS_timings.out
./cublas_benchmark 123 14 100 1 64 0 0 >> PbS_timings.out
./cublas_benchmark 123 15 100 1 64 0 0 >> PbS_timings.out
./cublas_benchmark 123 16 100 1 64 0 0 >> PbS_timings.out
./cublas_benchmark 123 17 100 1 64 0 0 >> PbS_timings.out
./cublas_benchmark 123 18 100 1 64 0 0 >> PbS_timings.out
./cublas_benchmark 123 19 100 1 64 0 0 >> PbS_timings.out
./cublas_benchmark 123 20 100 1 64 0 0 >> PbS_timings.out

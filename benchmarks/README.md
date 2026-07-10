# TAPP benchmarks

GPU benchmarks using the TAPP API. Each `*.cpp` is standalone; timing covers
only `TAPP_execute_product` (allocation/fill excluded). Storage and compute
datatypes are set near the top of each file.

```sh
cmake -B build -DTAPP_BENCHMARKS=ON -DTAPP_CUBLAS=ON \
      -DCUTT_ROOT=/path/to/cutt -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --target bench-ccsd_bottlenecks
./build/benchmarks/bench-ccsd_bottlenecks [nocc nvirt [nocc_PH nvirt_PH]]
```

Back-end: `tapp::cublas` if `TAPP_CUBLAS`, else `tapp::cutensor`. Add a benchmark
by dropping a source file here and listing it in `CMakeLists.txt`.

# How to run `benchmark.cu` script

Toy script to run benchmarks for PP/PH contractions with the following arguments:
- SEED: int
- NOCC: int
- NVIR: int
- PHI (double): for generating the elements of tensors with the distribution `uniform(0,1) e^{phi}`
- PREC_DIGITS (int): user-asked precision 
- N_REP (int): number of times the contraction is repeated for timing
- MODE (int): 0 for accuracy check/ 1 for timing only
- DTYPE_LENGTH (int): 32 for C32/ 64 for C64
- CONTRACTION (str): PP of PH
- EMULATION_STRAT_PERFORMANT (int): 0 for EAGER, 1 for PERFORMANT
- EMULATION_MANTISSA (int): 0 for FIXED, 1 for DYNAMIC
- ONLY_REAL_PART (int): 1 for ONLY_REAL_PART, 0 for generating general complex numbers

Generating executables:
```{shell}
make cutensor
make cublas
```

Running examples:
```
./cublas_benchmark 123 100 140 0.0 10 20 1 64 PP 0 0 1
./cutensor_benchmark 123 100 140 0.0 10 20 1 64 PP 1 1 0
```

Note: you will need to:
- modify the include/lib paths
- add the lib paths to LD_LIBRARY_PATH env variable


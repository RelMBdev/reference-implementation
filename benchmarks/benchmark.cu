#include "utils.h"
#include <tapp.h>

static void set_values(int64_t e[4], int64_t v1, int64_t v2, int64_t v3, int64_t v4)
{
	e[0] = v1;
	e[1] = v2;
	e[2] = v3;
	e[3] = v4;
}

static void strides4(const int64_t e[4], int64_t s[4])
{
    s[0] = 1; s[1] = e[0]; s[2] = e[0] * e[1]; s[3] = e[0] * e[1] * e[2];
}



template<typename T, typename TC>
int run_benchmark(int64_t SEED, int64_t NOCC, int64_t NVIR, T PHI, int64_t PREC_DIGITS, int64_t N_REP, int MODE, const char* CONTRACTION){

    printf("Starting benchmark with contraction %s\n", CONTRACTION);
    static TAPP_prectype PREC = TAPP_DEFAULT_PREC;

    printf("PREC DIGITS = %d\n", PREC_DIGITS);
    switch(PREC_DIGITS){
	    case 5:
		    PREC = TAPP_C_5_DIGITS;
		    break;
	    case 10:
		    PREC = TAPP_C_10_DIGITS;
		    break;
	    case 20:
		    PREC = TAPP_C_20_DIGITS;
		    break;
	    case 34:
		    PREC = TAPP_C_34_DIGITS;
		    break;
            default:
		    printf("PREC type not available\n");
    }

    TC ALPHA;
    TC BETA;

    if constexpr (sizeof(T)==sizeof(float)){
	    ALPHA = make_cuComplex(1.0, 0.0);
	    BETA = make_cuComplex(0.0, 0.0);
    }
    if constexpr (sizeof(T)==sizeof(double)){
	    ALPHA = make_cuDoubleComplex(1.0, 0.0);
	    BETA = make_cuDoubleComplex(0.0, 0.0);
    }

    int64_t nA, nB, nC, nD;
    int64_t eA[4];
    int64_t eB[4];
    int64_t eC[4];
    int64_t eD[4];
    int64_t iA[4];
    int64_t iB[4];
    int64_t iC[4];
    int64_t iD[4];
    bool is_PP = strcmp(CONTRACTION,"PP")==0;

    if(strcmp(CONTRACTION,"PP")==0){
	   printf("Setting up PP contraction...\n");
	   nA = static_cast<int64_t>(NVIR)*NVIR*NVIR*NVIR;
           nB = static_cast<int64_t>(NVIR)*NVIR*NOCC*NOCC;
           nC = static_cast<int64_t>(NVIR)*NVIR*NOCC*NOCC;
           nD = static_cast<int64_t>(NVIR)*NVIR*NOCC*NOCC;

	   set_values(eA, NVIR, NVIR, NVIR, NVIR);
           set_values(eB, NVIR, NVIR, NOCC, NOCC);
           set_values(eC, NVIR, NVIR, NOCC, NOCC);
           set_values(eD, NVIR, NVIR, NOCC, NOCC);

	   set_values(iA, 'a', 'b', 'c', 'd');
           set_values(iB, 'c', 'd', 'i', 'j');
           set_values(iC, 'a', 'b', 'i', 'j');
           set_values(iD, 'a', 'b', 'i', 'j');

    }
    else if(strcmp(CONTRACTION,"PH")==0){
	   printf("Setting up PH contraction...\n");
	   nA = static_cast<int64_t>(NVIR)*NVIR*NOCC*NOCC;
           nB = static_cast<int64_t>(NVIR)*NVIR*NOCC*NOCC;
           nC = static_cast<int64_t>(NVIR)*NVIR*NOCC*NOCC;
           nD = static_cast<int64_t>(NVIR)*NVIR*NOCC*NOCC;

	   set_values( eA, NOCC, NVIR, NVIR, NOCC);
           set_values( eB, NVIR, NVIR, NOCC, NOCC);
           set_values( eC, NVIR, NVIR, NOCC, NOCC);
           set_values( eD, NVIR, NVIR, NOCC, NOCC);

	   set_values( iA, 'k', 'b', 'c', 'j');
           set_values( iB, 'a', 'c', 'i', 'k');
           set_values( iC, 'a', 'b', 'i', 'j');
           set_values( iD, 'a', 'b', 'i', 'j');
    }
    else{
	    printf("CONTRACTION NOT RECOGNIZED...\n");
	    return 1;
    }

    int64_t sA[4], sB[4], sD[4];
    strides4(eA, sA); strides4(eB, sB); strides4(eD, sD);

    // Generate Random matrices
    void *devA, *devB, *devC, *devD;
    CUDA_CHECK(cudaMalloc(&devA, nA * sizeof(TC)));
    CUDA_CHECK(cudaMalloc(&devB, nB * sizeof(TC)));
    CUDA_CHECK(cudaMalloc(&devC, nC * sizeof(TC)));
    CUDA_CHECK(cudaMalloc(&devD, nD * sizeof(TC)));

    generate_random_matrix<T,TC>((TC*) devA, nA, PHI, SEED);
    generate_random_matrix<T,TC>((TC*) devB, nB, PHI, SEED+1);
    printf("Finished generating tensors\n");

    // GPU Result with TAPP and given backend
    TAPP_handle handle; TAPP_create_handle(&handle);
    TAPP_tensor_info Ainfo, Binfo, Cinfo, Dinfo;
    TAPP_create_tensor_info(&Ainfo, handle, (sizeof(T)==sizeof(float))? TAPP_C32 : TAPP_C64, 4, eA, sA);
    TAPP_create_tensor_info(&Binfo, handle, (sizeof(T)==sizeof(float))? TAPP_C32 : TAPP_C64, 4, eB, sB);
    TAPP_create_tensor_info(&Cinfo, handle, (sizeof(T)==sizeof(float))? TAPP_C32 : TAPP_C64, 4, eD, sD);
    TAPP_create_tensor_info(&Dinfo, handle, (sizeof(T)==sizeof(float))? TAPP_C32 : TAPP_C64, 4, eD, sD);

    TAPP_tensor_product plan;
    TAPP_create_tensor_product(&plan, handle,
        TAPP_IDENTITY, Ainfo, iA, TAPP_IDENTITY, Binfo, iB,
        TAPP_IDENTITY, Cinfo, iC, TAPP_IDENTITY, Dinfo, iD, PREC);

    TAPP_executor exec; TAPP_create_executor(&exec);
    TAPP_status status;
    TAPP_execute_product(plan, exec, &status, &ALPHA, devA, devB, &BETA, devC, devD); // warm-up
    cudaDeviceSynchronize();

    printf("Finished TAPP product\n");

    if(MODE==0){
    	// Comparison with CPU reference with Error metrics
    	TC* A = nullptr;
    	TC* B = nullptr;
    	TC* C = nullptr;
    	TC* D = nullptr;

    	CUDA_CHECK(cudaMallocHost(&A, nA * sizeof(TC)));
    	CUDA_CHECK(cudaMallocHost(&B, nB * sizeof(TC)));
    	CUDA_CHECK(cudaMallocHost(&C, nD * sizeof(TC)));
    	CUDA_CHECK(cudaMallocHost(&D, nD * sizeof(TC)));

    	CUDA_CHECK(cudaMemcpy(A, devA, nA * sizeof(TC), cudaMemcpyDeviceToHost));
    	CUDA_CHECK(cudaMemcpy(B, devB, nB * sizeof(TC), cudaMemcpyDeviceToHost));
    	CUDA_CHECK(cudaMemcpy(D, devD, nD * sizeof(TC), cudaMemcpyDeviceToHost));

	if(not is_PP){
    		#pragma omp parallel for collapse(4)
    		for(size_t a=0; a<NVIR; a++){
    		for(size_t b=0; b<NVIR; b++){
    		   for(size_t i=0; i<NOCC; i++){
    		   for(size_t j=0; j<NOCC; j++){
    		      TC contraction;
    		     
    		      if constexpr(sizeof(T)==sizeof(float)) contraction = make_cuComplex(0.0, 0.0);
    		      if constexpr(sizeof(T)==sizeof(double)) contraction = make_cuDoubleComplex(0.0, 0.0);

    		      for(size_t c=0; c<NVIR; c++){
    		      for(size_t k=0; k<NOCC; k++){
    		         size_t idA = k + b*NOCC + c*NOCC*NVIR + j*NOCC*NVIR*NVIR;
    		         size_t idB = a + c*NVIR + i*NVIR*NVIR + k*NVIR*NVIR*NOCC;
    		         if constexpr(sizeof(T)==sizeof(double)) contraction = cuCadd(
    		            contraction,
    		            cuCmul(A[idA], B[idB])
    		         );
    		         if constexpr(sizeof(T)==sizeof(float)) contraction = cuCaddf(
    		            contraction,
    		            cuCmulf(A[idA], B[idB])
    		         );
    		      }
    		      }
    		      size_t idC = a + b*NVIR + i*NVIR*NVIR + j*NVIR*NVIR*NOCC;
    		      if constexpr(sizeof(T)==sizeof(double)) C[idC] = cuCmul(ALPHA, contraction);
    		      if constexpr(sizeof(T)==sizeof(float)) C[idC] = cuCmulf(ALPHA, contraction);
    		   }
    		   }
    		}
    		}
	} else{
		#pragma omp parallel for collapse(4)
   		for(size_t a=0; a<NVIR; a++){
   		for(size_t b=0; b<NVIR; b++){
   		   for(size_t i=0; i<NOCC; i++){
   		   for(size_t j=0; j<NOCC; j++){
    		      TC contraction;
    		      if constexpr(sizeof(T)==sizeof(float)) contraction = make_cuComplex(0.0, 0.0);
    		      if constexpr(sizeof(T)==sizeof(double)) contraction = make_cuDoubleComplex(0.0, 0.0);

   		      for(size_t c=0; c<NVIR; c++){
   		      for(size_t d=0; d<NVIR; d++){
   		         size_t idA = a + b*NVIR + c*NVIR*NVIR + d*NVIR*NVIR*NVIR;
   		         size_t idB = c + d*NVIR + i*NVIR*NVIR + j*NVIR*NVIR*NOCC;
    		         if constexpr(sizeof(T)==sizeof(double)) contraction = cuCadd(
    		            contraction,
    		            cuCmul(A[idA], B[idB])
    		         );
    		         if constexpr(sizeof(T)==sizeof(float)) contraction = cuCaddf(
    		            contraction,
    		            cuCmulf(A[idA], B[idB])
    		         );
   		      }
   		      }
   		      size_t idC = a + b*NVIR + i*NVIR*NVIR + j*NVIR*NVIR*NOCC;
    		      if constexpr(sizeof(T)==sizeof(double)) C[idC] = cuCmul(ALPHA, contraction);
    		      if constexpr(sizeof(T)==sizeof(float)) C[idC] = cuCmulf(ALPHA, contraction);
   		   }
   		   }
   		}
   		}

	}

    	CUDA_CHECK(cudaMemcpy(devC, C, nC * sizeof(TC), cudaMemcpyHostToDevice));
    	printf("Finished CPU reference contraction!\n");

    	T max_rel_error = get_max_relative_error<T>((T*) devC, (T*) devD, nD*2);
    	T max_abs_error = get_max_absolute_error<T>((T*) devC, (T*) devD, nD*2);
    	printf("Max rel. error = %e\n", max_rel_error);
    	printf("Max abs. error = %e\n", max_abs_error);

	for(int i=0;i<10;i++){
		printf("%d) %f %f %f %f %f %f %f %f\n", i, A[i].x, A[i].y, B[i].x, B[i].y, C[i].x, C[i].y, D[i].x, D[i].y);
        }
    	cudaFreeHost(A); cudaFreeHost(B); cudaFreeHost(D);
    	cudaFree(devA); cudaFree(devB); cudaFree(devC); cudaFree(devD);

    	TAPP_destroy_tensor_product(plan);
    	TAPP_destroy_tensor_info(Ainfo); TAPP_destroy_tensor_info(Binfo);
    	TAPP_destroy_tensor_info(Cinfo); TAPP_destroy_tensor_info(Dinfo);
    	TAPP_destroy_executor(exec); TAPP_destroy_handle(handle);

    	return 0;
    }

    // Timings
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));

    for(int i=0;i<N_REP;i++){
        TAPP_execute_product(plan, exec, &status, &ALPHA, devA, devB, &BETA, devC, devD); // warm-up
        cudaDeviceSynchronize();
    }

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Finished cuTENSOR product and it took %f ms (average over %u repetitions)\n", ms/N_REP, N_REP);

    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(Ainfo); TAPP_destroy_tensor_info(Binfo);
    TAPP_destroy_tensor_info(Cinfo); TAPP_destroy_tensor_info(Dinfo);
    TAPP_destroy_executor(exec); TAPP_destroy_handle(handle);

    cudaFree(devA); cudaFree(devB); cudaFree(devC); cudaFree(devD);
    return 0;
}

int main(int argc, char const* argv[])
{
    // Read arguments: nocc, nvirt, phi, prec digits, n_repetition, type of execution (check-only, timing)
    const int64_t SEED = std::stoi(argv[1]);
    const int64_t NOCC = std::stoi(argv[2]);
    const int64_t NVIR = std::stoi(argv[3]);
    const double PHI  = std::stod(argv[4]);
    const int64_t PREC_DIGITS = std::stoi(argv[5]);
    const int64_t N_REP = std::stoi(argv[6]);
    const int MODE = std::stoi(argv[7]);
    const int64_t DTYPE_LENGTH = std::stoi(argv[8]);
    const char* CONTRACTION = argv[9];

    if(DTYPE_LENGTH==32) return run_benchmark<float, cuComplex> (SEED, NOCC, NVIR, static_cast<float>(PHI), PREC_DIGITS, N_REP, MODE, CONTRACTION);
    if(DTYPE_LENGTH==64) return run_benchmark<double, cuDoubleComplex>(SEED, NOCC, NVIR, PHI, PREC_DIGITS, N_REP, MODE, CONTRACTION);
    return 1;
}


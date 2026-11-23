#include "utils.h"
#include <tapp.h>
#include <attributes.h>

const uint32_t ALIGNMENT_REQUIREMENT = 128;

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

static void strides2(const int64_t e[2], int64_t s[2])
{
    s[0] = 1; s[1] = e[0];
}

template<typename T, typename TC>
int run_benchmark(int64_t SEED, int64_t PREC_DIGITS, int64_t N_REP, int MODE, bool EMULATION_STRATEGY_PERFORMANT, bool EMULATION_MANTISSA_CONTROL_DYNAMIC, bool RANDOM){

    static TAPP_prectype PREC = TAPP_DEFAULT_PREC;

    printf("PREC DIGITS = %d\n", PREC_DIGITS);
    switch(PREC_DIGITS){
       case -1:
          PREC = TAPP_CUTENSOR_C64_8XINT8_EMULATION;
          break;
	    case 3:
		    PREC = TAPP_C_3_DIGITS;
		    break;
	    case 4:
		    PREC = TAPP_C_4_DIGITS;
		    break;
	    case 5:
		    PREC = TAPP_C_5_DIGITS;
		    break;
	    case 6:
		    PREC = TAPP_C_6_DIGITS;
		    break;
	    case 7:
		    PREC = TAPP_C_7_DIGITS;
		    break;
	    case 8:
		    PREC = TAPP_C_8_DIGITS;
		    break;
	    case 9:
		    PREC = TAPP_C_9_DIGITS;
		    break;
	    case 10:
		    PREC = TAPP_C_10_DIGITS;
		    break;
	    case 11:
		    PREC = TAPP_C_11_DIGITS;
		    break;
	    case 12:
		    PREC = TAPP_C_12_DIGITS;
		    break;
	    case 13:
		    PREC = TAPP_C_13_DIGITS;
		    break;
	    case 14:
		    PREC = TAPP_C_14_DIGITS;
		    break;
	    case 15:
		    PREC = TAPP_C_15_DIGITS;
		    break;
	    case 16:
		    PREC = TAPP_C_16_DIGITS;
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

    cuDoubleComplex ONE = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex ZERO = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex ONE_QUARTER = make_cuDoubleComplex(0.25, 0.0);

    TC ONE_T;
    TC ZERO_T;
    TC ONE_QUARTER_T;
    if constexpr (sizeof(T)==sizeof(float)){
	    ONE_T = make_cuComplex(1.0, 0.0);
	    ZERO_T = make_cuComplex(0.0, 0.0);
	    ONE_QUARTER_T = make_cuComplex(0.25, 0.0);
    }
    if constexpr (sizeof(T)==sizeof(double)){
	    ONE_T = make_cuDoubleComplex(1.0, 0.0);
	    ZERO_T = make_cuDoubleComplex(0.0, 0.0);
	    ONE_QUARTER_T = make_cuDoubleComplex(0.25, 0.0);
    }

    // GPU Result with TAPP and given backend
    TAPP_handle handle; TAPP_create_handle(&handle);
#if USE_CUBLAS
    // Resolve values with function-level scope (guaranteed lifetime)
    int strategy_val = EMULATION_STRATEGY_PERFORMANT
        ? TAPP_EMULATION_STRATEGY_PERFORMANT
        : TAPP_EMULATION_STRATEGY_EAGER;
    
    if (EMULATION_STRATEGY_PERFORMANT) {
        printf("EMULATION_STRATEGY_PERFORMANT\n");
    } else {
        printf("EMULATION_STRATEGY_EAGER\n");
    }
    
    TAPP_error err1 = TAPP_attr_set(handle, ATTR_KEY_EMULATION_STRATEGY, &strategy_val);
    
    int mantissa_val = EMULATION_MANTISSA_CONTROL_DYNAMIC
        ? TAPP_EMULATION_MANTISSA_CONTROL_DYNAMIC
        : TAPP_EMULATION_MANTISSA_CONTROL_FIXED;
    
    if (EMULATION_MANTISSA_CONTROL_DYNAMIC) {
        printf("EMULATION_MANTISSA_CONTROL_DYNAMIC\n");
    } else {
        printf("EMULATION_MANTISSA_CONTROL_FIXED\n");
    }
    
    TAPP_error err2 = TAPP_attr_set(handle, ATTR_KEY_EMULATION_MANTISSA_CONTROL, &mantissa_val);
#endif

    TAPP_error error;
    int message_len;
    char *message_buff;

    int64_t nvir = 400, nocc = 10;
    int64_t nA=nvir, nB=nvir, nI=nocc, nJ=nocc;

    printf("Before tensor info\n");
    TAPP_tensor_info Finfo, T1info, Vinfo, Tauinfo, Einfo;
    int64_t iF[2] = {'i', 'a'}; 
    int64_t eF[2] = {nI, nA}; 
    //int64_t sF[2] = { 1, nI}; 
    int64_t sF[2]; 
    strides2(eF, sF);
    size_t nF = nI*nA; 
    TAPP_create_tensor_info(&Finfo, handle, (sizeof(T)==sizeof(float))? TAPP_C32 : TAPP_C64, 2, eF, sF);
    int64_t iT1[2] = {'a', 'i'}; 
    int64_t eT1[2] = {nA, nI}; 
    //int64_t sT1[2] = { 1, nA}; 
    int64_t sT1[2]; 
    strides2(eT1, sT1);
    size_t nT1 = nI*nA; 
    TAPP_create_tensor_info(&T1info, handle, (sizeof(T)==sizeof(float))? TAPP_C32 : TAPP_C64, 2, eT1, sT1);
    int64_t iV[4] = {'i', 'j', 'a', 'b'}; 
    int64_t eV[4] = {nI, nJ, nA, nB}; 
    //int64_t sV[4] = {1, nI, nI*nJ, nI*nJ*nA}; 
    int64_t sV[4];
    strides4(eV, sV);
    size_t nV = nI*nJ*nA*nB; 
    TAPP_create_tensor_info(&Vinfo, handle, (sizeof(T)==sizeof(float))? TAPP_C32 : TAPP_C64, 4, eV, sV);
    int64_t iTau[4] = {'a', 'b', 'i', 'j'}; 
    int64_t eTau[4] = {nA, nB, nI, nJ}; 
    //int64_t sTau[4] = {1, nA, nA*nB, nA*nB*nI}; 
    int64_t sTau[4]; 
    strides4(eTau, sTau);
    size_t nTau = nI*nJ*nA*nB; 
    TAPP_create_tensor_info(&Tauinfo, handle, (sizeof(T)==sizeof(float))? TAPP_C32 : TAPP_C64, 4, eTau, sTau);
    int64_t iE[1] = {'x'}; 
    int64_t eE[1] = {1}; 
    int64_t sE[1] = {1}; 
    size_t nE = 1; 
    TAPP_create_tensor_info(&Einfo, handle, (sizeof(T)==sizeof(float))? TAPP_C32 : TAPP_C64, 0, eE, sE);

    printf("Before tensor plans\n");
    TAPP_tensor_product plan1;
    error = TAPP_create_tensor_product(&plan1, handle,
        TAPP_IDENTITY, Finfo, iF, TAPP_IDENTITY, T1info, iT1,
        TAPP_IDENTITY, Einfo, iE, TAPP_IDENTITY, Einfo, iE, PREC);
    if(!TAPP_check_success(error)){
        printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
        message_len = TAPP_explain_error(error, 0, NULL);
        message_buff = (char*) malloc((message_len + 1) * sizeof(char));
        TAPP_explain_error(error, message_len + 1, message_buff);
        if(!TAPP_check_success(error)) printf(message_buff);
       return 1;
    }

    TAPP_tensor_product plan2;
    error = TAPP_create_tensor_product(&plan2, handle,
        TAPP_IDENTITY, Vinfo, iV, TAPP_IDENTITY, Tauinfo, iTau,
        TAPP_IDENTITY, Einfo, iE, TAPP_IDENTITY, Einfo, iE, PREC);
    if(!TAPP_check_success(error)){
        printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
        message_len = TAPP_explain_error(error, 0, NULL);
        message_buff = (char*) malloc((message_len + 1) * sizeof(char));
        TAPP_explain_error(error, message_len + 1, message_buff);
        if(!TAPP_check_success(error)) printf(message_buff);
       return 1;
    }

    printf("Before tensor executor\n");
    TAPP_executor exec; 
    error = TAPP_create_executor(&exec);
    if(!TAPP_check_success(error)){
        printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
        message_len = TAPP_explain_error(error, 0, NULL);
        message_buff = (char*) malloc((message_len + 1) * sizeof(char));
        TAPP_explain_error(error, message_len + 1, message_buff);
        if(!TAPP_check_success(error)) printf(message_buff);
	
       return 1;
    }

    // Allocating data on CPU and GPU
    printf("Before tensor alloc on CPU\n");
    void *F=nullptr, *T1=nullptr, *V=nullptr, *Tau = nullptr;
    void *E = nullptr;
    void *E_ref = nullptr;

    CUDA_CHECK(cudaMallocHost(&F, nF * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMallocHost(&T1, nT1 * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMallocHost(&V, nV * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMallocHost(&Tau, nTau * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMallocHost(&E, nE * sizeof(cuDoubleComplex))); // TAPP result
    CUDA_CHECK(cudaMallocHost(&E_ref, nE * sizeof(cuDoubleComplex))); // cuTENSOR reference

    void *devF=nullptr, *devT1=nullptr, *devV=nullptr, *devTau = nullptr;
    void *devE = nullptr;
    void *devE_ref = nullptr;

    printf("Before tensor copy to GPU\n");
    CUDA_CHECK(cudaMalloc(&devF, nF * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&devT1, nT1 * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&devV, nV * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&devTau, nTau * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&devE, nE * sizeof(cuDoubleComplex))); // TAPP result
    CUDA_CHECK(cudaMalloc(&devE_ref, nE * sizeof(cuDoubleComplex))); // cuTENSOR reference

    if(not RANDOM){
    // Reading data from Hdf5 files
        printf("Before tensor read from HDF5 files\n");
        std::string name = "Fov.h5";
        read_hdf5((cuDoubleComplex *)F, nF, name);
        name = "t1.h5";
        read_hdf5((cuDoubleComplex *)T1, nT1, name);
        name = "V.h5";
        read_hdf5((cuDoubleComplex *)V, nV, name);
        name = "tau.h5";
        read_hdf5((cuDoubleComplex *)Tau, nTau, name); 

        CUDA_CHECK(cudaMemcpy(devF, F, nF * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devT1, T1, nT1 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devV, V, nV * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devTau, Tau, nTau * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    else{
        printf("Generate random tensors!\n");
        generate_random_matrix((cuDoubleComplex*) devF, nF, 3.0, SEED);
        generate_random_matrix((cuDoubleComplex*) devT1, nT1, 3.0, SEED+1);
        generate_random_matrix((cuDoubleComplex*) devV, nV, 3.0, SEED+1);
        generate_random_matrix((cuDoubleComplex*) devTau, nTau, 3.0, SEED+1);

        CUDA_CHECK(cudaMemcpy(F, devF, nF * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(T1, devT1, nT1 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(V, devV, nV * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Tau, devTau, nTau * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }

    // Downcasting if necessary
    printf("Before tensor copy to GPU for TAPP\n");
    void *devF_T=nullptr, *devT1_T=nullptr, *devV_T=nullptr, *devTau_T = nullptr;
    void *devE_T = nullptr;
    if constexpr (sizeof(T)==sizeof(float)){
            CUDA_CHECK(cudaMalloc(&devF_T, nF * sizeof(cuComplex)));
            CUDA_CHECK(cudaMalloc(&devT1_T, nT1 * sizeof(cuComplex)));
            CUDA_CHECK(cudaMalloc(&devV_T, nV * sizeof(cuComplex)));
            CUDA_CHECK(cudaMalloc(&devTau_T, nTau * sizeof(cuComplex)));
            CUDA_CHECK(cudaMalloc(&devE_T, nE * sizeof(cuComplex)));

	    complex_double_to_float((const cuDoubleComplex *) devF, (cuComplex *) devF_T, nF);
	    complex_double_to_float((const cuDoubleComplex *) devT1, (cuComplex *) devT1_T, nT1);
	    complex_double_to_float((const cuDoubleComplex *) devV, (cuComplex *) devV_T, nV);
	    complex_double_to_float((const cuDoubleComplex *) devTau, (cuComplex *) devTau_T, nTau);
    }
    if constexpr (sizeof(T)==sizeof(double)){
	    devF_T = devF;
	    devT1_T = devT1;
	    devV_T = devV;
	    devTau_T = devTau;
	    devE_T = devE;
    }

    // Actual TAPP tensor product
    printf("Before execute product 1\n");
    error = TAPP_execute_product(plan1, exec, 0, &ONE_T, devF_T, devT1_T, &ZERO_T, devE_T, devE_T);
    if(!TAPP_check_success(error)){
        printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
        message_len = TAPP_explain_error(error, 0, NULL);
        message_buff = (char*) malloc((message_len + 1) * sizeof(char));
        TAPP_explain_error(error, message_len + 1, message_buff);
        if(!TAPP_check_success(error)) printf(message_buff);
       return 1;
    }
    printf("Before execute product 2\n");
    error = TAPP_execute_product(plan2, exec, 0, &ONE_QUARTER_T, devV_T, devTau_T, &ONE_T, devE_T, devE_T);
    if(!TAPP_check_success(error)){
        printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
        message_len = TAPP_explain_error(error, 0, NULL);
        message_buff = (char*) malloc((message_len + 1) * sizeof(char));
        TAPP_explain_error(error, message_len + 1, message_buff);
        if(!TAPP_check_success(error)) printf(message_buff);
       return 1;
    }
    if constexpr (sizeof(T)==sizeof(float)){
            complex_float_to_double((const cuComplex *) devE_T, (cuDoubleComplex *)devE, nE);
            CUDA_CHECK(cudaFree(devF_T));
            CUDA_CHECK(cudaFree(devT1_T));
            CUDA_CHECK(cudaFree(devV_T));
            CUDA_CHECK(cudaFree(devTau_T));
            CUDA_CHECK(cudaFree(devE_T));
    }
    printf("Finished TAPP product\n");

    if(MODE==0){
        // cuTENSOR reference
	cutensorHandle_t cutensor_handle;
	CUTENSOR_CHECK(cutensorCreate(&cutensor_handle));

        cutensorTensorDescriptor_t descF;
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
            cutensor_handle,
            &descF,
            2,
            eF,
            sF,
            CUDA_C_64F,
            ALIGNMENT_REQUIREMENT
        ));

        cutensorTensorDescriptor_t descT1;
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
            cutensor_handle,
            &descT1,
            2,
            eT1,
            sT1,
            CUDA_C_64F,
            ALIGNMENT_REQUIREMENT
        ));

        cutensorTensorDescriptor_t descV;
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
            cutensor_handle,
            &descV,
            4,
            eV,
            sV,
            CUDA_C_64F,
            ALIGNMENT_REQUIREMENT
        ));

        cutensorTensorDescriptor_t descTau;
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
            cutensor_handle,
            &descTau,
            4,
            eTau,
            sTau,
            CUDA_C_64F,
            ALIGNMENT_REQUIREMENT
        ));

        cutensorTensorDescriptor_t descE;
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
            cutensor_handle,
            &descE,
            0,
            eE,
            sE,
            CUDA_C_64F,
            ALIGNMENT_REQUIREMENT
        ));

        std::vector<int32_t> iF_32 = std::vector<int32_t>(iF, iF + 2);
        std::vector<int32_t> iT1_32 = std::vector<int32_t>(iT1, iT1 + 2);
        std::vector<int32_t> iV_32 = std::vector<int32_t>(iV, iV + 4);
        std::vector<int32_t> iTau_32 = std::vector<int32_t>(iTau, iTau + 4);
        std::vector<int32_t> iE_32 = std::vector<int32_t>(iE, iE);

        cutensorPlanPreference_t pref;
	CUTENSOR_CHECK(cutensorCreatePlanPreference(
            cutensor_handle,
            &pref,
            CUTENSOR_ALGO_DEFAULT,
            CUTENSOR_JIT_MODE_NONE
        ));

	// Contraction 1 with F and T1
        cutensorOperationDescriptor_t descContract1;
	CUTENSOR_CHECK(cutensorCreateContraction(
            cutensor_handle,
            &descContract1,
            descF,
            iF_32.data(),
            CUTENSOR_OP_IDENTITY,
            descT1,
            iT1_32.data(),
            CUTENSOR_OP_IDENTITY,
            descE,
            iE_32.data(),
            CUTENSOR_OP_IDENTITY,
            descE,
            iE_32.data(),
            CUTENSOR_COMPUTE_DESC_64F
        ));

        uint64_t workspaceSizeEstimate1;
        CUTENSOR_CHECK(cutensorEstimateWorkspaceSize(
           cutensor_handle,
           descContract1,
           pref,
           CUTENSOR_WORKSPACE_DEFAULT,
           &workspaceSizeEstimate1
       ));

        cutensorPlan_t planContract1;
        CUTENSOR_CHECK(cutensorCreatePlan(
          cutensor_handle,
          &planContract1,
          descContract1,
          pref,
          workspaceSizeEstimate1
        ));

	// Contraction 2 with V and Tau
        cutensorOperationDescriptor_t descContract2;
	CUTENSOR_CHECK(cutensorCreateContraction(
            cutensor_handle,
            &descContract2,
            descV,
            iV_32.data(),
            CUTENSOR_OP_IDENTITY,
            descTau,
            iTau_32.data(),
            CUTENSOR_OP_IDENTITY,
            descE,
            iE_32.data(),
            CUTENSOR_OP_IDENTITY,
            descE,
            iE_32.data(),
            CUTENSOR_COMPUTE_DESC_64F
        ));

        uint64_t workspaceSizeEstimate2;
        CUTENSOR_CHECK(cutensorEstimateWorkspaceSize(
           cutensor_handle,
           descContract2,
           pref,
           CUTENSOR_WORKSPACE_DEFAULT,
           &workspaceSizeEstimate2
       ));

        cutensorPlan_t planContract2;
        CUTENSOR_CHECK(cutensorCreatePlan(
          cutensor_handle,
          &planContract2,
          descContract2,
          pref,
          workspaceSizeEstimate2
        ));

	void* workspace;
        CUDA_CHECK(cudaMalloc(&workspace, std::max(workspaceSizeEstimate1,workspaceSizeEstimate2)));

	// Actual cuTENSOR calculation
	cutensorContract(
           cutensor_handle,
           planContract1,
           &ONE,
           devF,
           devT1,
           &ZERO,
           devE_ref,
           devE_ref,
           workspace,
           std::max(workspaceSizeEstimate1,workspaceSizeEstimate2),
           0
       );

        void *devE_copy = nullptr;
        CUDA_CHECK(cudaMalloc(&devE_copy, nE * sizeof(cuDoubleComplex))); // TAPP result
        CUDA_CHECK(cudaMemcpy(devE_copy, devE_ref, nE * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

	cutensorContract(
           cutensor_handle,
           planContract2,
           &ONE_QUARTER,
           devV,
           devTau,
           &ONE,
           devE_copy,
           devE_ref,
           workspace,
           std::max(workspaceSizeEstimate1,workspaceSizeEstimate2),
           0
       );

	cutensorDestroyTensorDescriptor(descF);
	cutensorDestroyTensorDescriptor(descT1);
	cutensorDestroyTensorDescriptor(descV);
	cutensorDestroyTensorDescriptor(descTau);
	cutensorDestroyTensorDescriptor(descE);
	cutensorDestroyPlanPreference(pref);
	cutensorDestroyOperationDescriptor(descContract1);
	cutensorDestroyPlan(planContract1);
	cutensorDestroyOperationDescriptor(descContract2);
	cutensorDestroyPlan(planContract2);
	cutensorDestroy(cutensor_handle);

	// Comparing results with 
        CUDA_CHECK(cudaMemcpy(E, devE, nE * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(E_ref, devE_ref, nE * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

	cuDoubleComplex* E_cast = static_cast<cuDoubleComplex*>(E);
	cuDoubleComplex* E_ref_cast = static_cast<cuDoubleComplex*>(E_ref);
	cuDoubleComplex E_dirac = make_cuDoubleComplex(-0.258738050783954, 0.f);

	printf("Error:\n");
	printf("Max rel. error = %e\n", cuCabs(cuCsub(E_cast[0],E_ref_cast[0]))/cuCabs(E_ref_cast[0]));
	printf("Max abs. error = %e\n", cuCabs(cuCsub(E_cast[0],E_ref_cast[0])));
	printf("Max rel. error with DIRAC C64 = %e\n", cuCabs(cuCsub(E_dirac,E_ref_cast[0]))/cuCabs(E_ref_cast[0]));
	printf("Max abs. error with DIRAC C64 = %e\n", cuCabs(cuCsub(E_dirac,E_ref_cast[0])));

	printf("Values:\n");
	printf("TAPP result: (%e, %e)\n", E_cast[0].x, E_cast[0].y);
	printf("cuTENSOR result: (%e, %e)\n", E_ref_cast[0].x, E_ref_cast[0].y);
	printf("DIRAC result: %e\n", -0.258738050783954);
    }
    else{
        // Warmup
        for(int i=0;i<10;i++){
             TAPP_execute_product(plan1, exec, 0, &ONE_T, devF_T, devT1_T, &ZERO_T, devE_T, devE_T);
             TAPP_execute_product(plan2, exec, 0, &ONE_QUARTER_T, devV_T, devTau_T, &ONE_T, devE_T, devE_T);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timings
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, 0));

        for(int i=0;i<N_REP;i++){
             TAPP_execute_product(plan1, exec, 0, &ONE_T, devF_T, devT1_T, &ZERO_T, devE_T, devE_T);
             TAPP_execute_product(plan2, exec, 0, &ONE_QUARTER_T, devV_T, devTau_T, &ONE_T, devE_T, devE_T);
        }

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("Finished cuTENSOR product and it took %f ms (average over %u repetitions)\n", ms/N_REP, N_REP);

    }

    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_info(Finfo);
    TAPP_destroy_tensor_info(T1info);
    TAPP_destroy_tensor_info(Vinfo);
    TAPP_destroy_tensor_info(Tauinfo);
    TAPP_destroy_tensor_info(Einfo);
    TAPP_destroy_tensor_product(plan1);
    TAPP_destroy_tensor_product(plan2);
    TAPP_destroy_executor(exec);

    CUDA_CHECK(cudaFree(devF));
    CUDA_CHECK(cudaFree(devT1));
    CUDA_CHECK(cudaFree(devV));
    CUDA_CHECK(cudaFree(devTau));
    CUDA_CHECK(cudaFree(devE)); // TAPP result
    CUDA_CHECK(cudaFree(devE_ref)); // cuTENSOR reference

    CUDA_CHECK(cudaFreeHost(F));
    CUDA_CHECK(cudaFreeHost(T1));
    CUDA_CHECK(cudaFreeHost(V));
    CUDA_CHECK(cudaFreeHost(Tau));
    CUDA_CHECK(cudaFreeHost(E)); // TAPP result
    CUDA_CHECK(cudaFreeHost(E_ref)); // cuTENSOR reference

    return 0;
}

int main(int argc, char const* argv[])
{
    // Read arguments: nocc, nvirt, phi, prec digits, n_repetition, type of execution (check-only, timing)
    const int64_t SEED = std::stoi(argv[1]);
    const int64_t PREC_DIGITS = std::stoi(argv[2]);
    const int64_t N_REP = std::stoi(argv[3]);
    const int MODE = std::stoi(argv[4]);
    const int64_t DTYPE_LENGTH = std::stoi(argv[5]);
    const int EMULATION_STRATEGY_PERFORMANT_INT = std::stoi(argv[6]);
    const int EMULATION_MANTISSA_CONTROL_DYNAMIC_INT = std::stoi(argv[7]);
    const int RANDOM_INT = std::stoi(argv[8]);

    const bool EMULATION_STRATEGY_PERFORMANT = EMULATION_STRATEGY_PERFORMANT_INT==1;
    const bool EMULATION_MANTISSA_CONTROL_DYNAMIC = EMULATION_MANTISSA_CONTROL_DYNAMIC_INT==1;
    const bool RANDOM = RANDOM_INT==1;

    if(DTYPE_LENGTH==32) return run_benchmark<float, cuComplex> (SEED, PREC_DIGITS, N_REP, MODE, EMULATION_STRATEGY_PERFORMANT, EMULATION_MANTISSA_CONTROL_DYNAMIC, RANDOM);
    if(DTYPE_LENGTH==64) return run_benchmark<double, cuDoubleComplex>(SEED, PREC_DIGITS, N_REP, MODE, EMULATION_STRATEGY_PERFORMANT, EMULATION_MANTISSA_CONTROL_DYNAMIC, RANDOM);
    return 1;
}


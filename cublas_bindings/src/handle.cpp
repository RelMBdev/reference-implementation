#include "../include/handle.h"
#include "../include/attributes.h"
#include "datatype.h"

TAPP_error TAPP_create_handle(TAPP_handle* handle)
{
    struct handle* handle_struct = new struct handle;
    cublasStatus_t stat = cublasCreate(&handle_struct->cublas);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        delete handle_struct;
        return tapp_error(stat);
    }
    handle_struct->attributes = new intptr_t[ATTR_COUNT];
    handle_struct->attributes[ATTR_KEY_USE_DEVICE_MEMORY] = (intptr_t) new bool(true);
    handle_struct->attributes[ATTR_KEY_EMULATION_STRATEGY] = (intptr_t) new int(TAPP_DEFAULT_EMULATION_STRATEGY);
    handle_struct->attributes[ATTR_KEY_EMULATION_MANTISSA_CONTROL] = (intptr_t) new int(TAPP_DEFAULT_EMULATION_MANTISSA_CONTROL);
    handle_struct->attributes[ATTR_KEY_EMULATION_NUM_GEMM] = (intptr_t) new int(TAPP_DEFAULT_NUM_GEMM);
    *handle = (TAPP_handle) handle_struct;
    return TAPP_SUCCESS;
}

TAPP_error TAPP_destroy_handle(TAPP_handle handle)
{
    struct handle* handle_struct = (struct handle*) handle;
    cublasStatus_t stat = cublasDestroy(handle_struct->cublas);
    delete (bool*)handle_struct->attributes[ATTR_KEY_USE_DEVICE_MEMORY];
    delete (int*)handle_struct->attributes[ATTR_KEY_EMULATION_STRATEGY];
    delete (int*)handle_struct->attributes[ATTR_KEY_EMULATION_MANTISSA_CONTROL];
    delete (int*) handle_struct->attributes[ATTR_KEY_EMULATION_NUM_GEMM];
    delete[] handle_struct->attributes;
    delete handle_struct;
    if (stat != CUBLAS_STATUS_SUCCESS) return tapp_error(stat);
    return TAPP_SUCCESS;
}

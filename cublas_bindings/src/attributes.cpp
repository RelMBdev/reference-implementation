#include "../include/attributes.h"

// Number of bytes stored for each attribute key.
static size_t attr_size(TAPP_key key)
{
    switch (key)
    {
    case ATTR_KEY_USE_DEVICE_MEMORY: return sizeof(bool);
    case ATTR_KEY_EMULATION_STRATEGY: return sizeof(int);
    case ATTR_KEY_EMULATION_MANTISSA_CONTROL: return sizeof(int);
    case ATTR_KEY_EMULATION_NUM_GEMM: return sizeof(int);
    default:                         return 0;
    }
}

TAPP_error TAPP_attr_set(TAPP_attr attr, TAPP_key key, void* value)
{
    struct handle* handle_struct = (struct handle*) attr;
    size_t size = attr_size(key);
    if (size == 0) return tapp_error(TAPP_ERROR_TYPE_TAPP, TAPP_ERR_INVALID_KEY);
    memcpy((void*)handle_struct->attributes[key], value, size);
    return TAPP_SUCCESS;
}

TAPP_error TAPP_attr_get(TAPP_attr attr, TAPP_key key, void** value)
{
    struct handle* handle_struct = (struct handle*) attr;
    size_t size = attr_size(key);
    if (size == 0) return tapp_error(TAPP_ERROR_TYPE_TAPP, TAPP_ERR_INVALID_KEY);
    *value = (void*)handle_struct->attributes[key];
    return TAPP_SUCCESS;
}

TAPP_error TAPP_attr_clear(TAPP_attr attr, TAPP_key key)
{
    struct handle* handle_struct = (struct handle*) attr;
    switch (key)
    {
    case ATTR_KEY_USE_DEVICE_MEMORY:
        *(bool*)handle_struct->attributes[key] = false;
        break;
    case ATTR_KEY_EMULATION_STRATEGY:
        *(int*)handle_struct->attributes[key] = 0;
        break;
    case ATTR_KEY_EMULATION_MANTISSA_CONTROL:
        *(int*)handle_struct->attributes[key] = 0;
        break;
    case ATTR_KEY_EMULATION_NUM_GEMM:
        *(int*)handle_struct->attributes[key] = 0;
        break;
    default:
        return tapp_error(TAPP_ERROR_TYPE_TAPP, TAPP_ERR_INVALID_KEY);
    }
    return TAPP_SUCCESS;
}

#ifndef TAPP_DATATYPE_H_
#define TAPP_DATATYPE_H_

#include "util.h"

/*
 * Storage data types:
 *
 * The storage data type is an integer enumeration which indicates the numerical format of tensor data as passed
 * into or returned from the library. Each tensor has a single storage datatype which describes all elements of
 * the tensor. Negative values and values less than 0x1000 are reserved by the standard, but values greater than
 * or equal to 0x1000 may be used by implementations for additional data types.
 */

typedef int TAPP_datatype;

enum
{
    /* IEEE754 float32: 1 sign bit, 8 exponent bits, 23 explicit significand bits */
    TAPP_F32 = 0,

    /* IEEE754 float64: 1 sign bit, 11 exponent bits, 52 explicit significand bits */
    TAPP_F64 = 1,

    /* Complex IEEE754 float32, stored with consecutive real and imaginary parts packed into 8 bytes */
    TAPP_C32 = 2,

    /* Complex IEEE754 float64, stored with consecutive real and imaginary parts packed into 16 bytes */
    TAPP_C64 = 3,

    /* IEEE754 float16: 1 sign bit, 5 exponent bits, 10 explicit significand bits */
    TAPP_F16 = 4,

    /* bfloat16: 1 sign bit, 8 exponent bits, 7 explicit significand bits */
    TAPP_BF16 = 5,

    /* Aliases */
    TAPP_FLOAT = TAPP_F32,
    TAPP_DOUBLE = TAPP_F64,
    TAPP_SCOMPLEX = TAPP_C32,
    TAPP_DCOMPLEX = TAPP_C64,
};

/*
 * Computational precision types:
 *
 * The computational precision determines the number of correct significant digits in the multiplication and
 * accumulation of scalar floating-point types. The computational precision also typically determines the conversion
 * of data to/from storage data types into an internal representation which may or may not be another storage data
 * type. The names of the precision type values are of the form TAPP_XXXYYY_ACCUM_ZZZ, where XXX and YYY indicate
 * the precision of the input scalars before multiplication, and ZZZ indicates the precision of the product after
 * accumulation. Note that when fused-multiply-add (FMA) instructions are available, the precision of the intermediate
 * product is "infinite". Low-precision computations, e.g. float16, may be performed in a higher precision when
 * hardware support is not available. The default precision TAPP_DEFAULT_PREC indicates that a computational precision
 * should be used which is not less than that of any of the input or output operands, but not typically greater than
 * that of any input or output operand (except for low-precision types as noted). For example, the multiplication
 * of float32 and float64 scalars accumulated into a float64 scalar could be performed in either f32f32_accum_f32
 * or f64f64_accum_f64 precision. The first option would require conversion of float64->float32 for one input operand
 * and float32->float64 during accumulation, while the second option would require float32->float64 conversion of the
 * other input operand.
 *
 * The computational precision indicates the precision of accumulation for a single scalar product, which may or may
 * not be maintained accross a large number of scalar operations. Any additional accumulation, e.g. from registers
 * to memory, or additional scalar multiplications such as multiplication by beta in a tensor contraction will be
 * performed in a precision greater than or equal to the precision of the storage data type of the output operand,
 * even if this precision is less than the indicated computational precision. For example, register-to-memory
 * accumulation for an output of type float16 may use float16 arithmetic even if the computational precision is
 * f16f16_accum_f32.
 */

typedef int TAPP_prectype;

enum
{
    /* The computational precision is determined as *at least* the lowest precision of the storage data types */
    /* of all input and output operands. */
    TAPP_DEFAULT_PREC = -1,

    /* IEEE754 float32 with or without singly-rounded FMA */
    TAPP_F32F32_ACCUM_F32 = TAPP_F32, /* = 0 */

    /* IEEE754 float64 with or without singly-rounded FMA */
    TAPP_F64F64_ACCUM_F64 = TAPP_F64, /* = 1 */

    /* IEEE754 float16 with or without singly-rounded FMA */
    /* Implementations may compute final or intermediate results in a higher precision */
    TAPP_F16F16_ACCUM_F16 = TAPP_F16, /* = 3 */

    /* float16 with wide accumulation */
    /* Implementations may compute intermediate results in a higher precision */
    TAPP_F16F16_ACCUM_F32 = 5,

    /* bfloat16 with wide accumulation */
    TAPP_BF16BF16_ACCUM_F32 = 6,

    /*
     * Variable-precision compute types for fixed-point / Ozaki-style FP64
     * emulation (e.g. cuBLAS emulated fixed-point, or a dedicated Ozaki
     * back-end). The integer in the name is the number of correct decimal
     * digits of precision requested in the accumulation. The TAPP_F_* values
     * are for real outputs, TAPP_C_* for complex. They cover 3..16 digits plus
     * a 34-digit (quad-like) point. The values are encoded so the digit count
     * is recoverable as (value % 1000): real are 1000 + digits, complex are
     * 2000 + digits.
     */
    TAPP_F_3_DIGITS  = 1003,
    TAPP_F_4_DIGITS  = 1004,
    TAPP_F_5_DIGITS  = 1005,
    TAPP_F_6_DIGITS  = 1006,
    TAPP_F_7_DIGITS  = 1007,
    TAPP_F_8_DIGITS  = 1008,
    TAPP_F_9_DIGITS  = 1009,
    TAPP_F_10_DIGITS = 1010,
    TAPP_F_11_DIGITS = 1011,
    TAPP_F_12_DIGITS = 1012,
    TAPP_F_13_DIGITS = 1013,
    TAPP_F_14_DIGITS = 1014,
    TAPP_F_15_DIGITS = 1015,
    TAPP_F_16_DIGITS = 1016,
    TAPP_F_34_DIGITS = 1034,

    TAPP_C_3_DIGITS  = 2003,
    TAPP_C_4_DIGITS  = 2004,
    TAPP_C_5_DIGITS  = 2005,
    TAPP_C_6_DIGITS  = 2006,
    TAPP_C_7_DIGITS  = 2007,
    TAPP_C_8_DIGITS  = 2008,
    TAPP_C_9_DIGITS  = 2009,
    TAPP_C_10_DIGITS = 2010,
    TAPP_C_11_DIGITS = 2011,
    TAPP_C_12_DIGITS = 2012,
    TAPP_C_13_DIGITS = 2013,
    TAPP_C_14_DIGITS = 2014,
    TAPP_C_15_DIGITS = 2015,
    TAPP_C_16_DIGITS = 2016,
    TAPP_C_20_DIGITS = 2020,
    TAPP_C_34_DIGITS = 2034,
};

/* Types used for Ozaki-I scheme:
 * - cuBLAS emulation strategy: eager vs performant
 * - cuBLAS emulation mantissa: dynamic vs fixed
 */
typedef int TAPP_EMULATION_STRATEGY;

enum
{
    TAPP_EMULATION_STRATEGY_PERFORMANT,
    TAPP_EMULATION_STRATEGY_EAGER,
    TAPP_DEFAULT_EMULATION_STRATEGY = TAPP_EMULATION_STRATEGY_PERFORMANT,
};

typedef int TAPP_EMULATION_MANTISSA_CONTROL;

enum
{
    TAPP_EMULATION_MANTISSA_CONTROL_DYNAMIC,
    TAPP_EMULATION_MANTISSA_CONTROL_FIXED,
    TAPP_DEFAULT_EMULATION_MANTISSA_CONTROL = TAPP_EMULATION_MANTISSA_CONTROL_DYNAMIC,
};

/* Type used for Ozaki-II scheme. It dictates the number of splits, which is also equal to the number of executed GEMMs.
 * There are not clear relation between the number of splits and the accuracy of the GEMM.
 */
typedef int TAPP_num_gemm;

enum
{
    TAPP_DEFAULT_NUM_GEMM = 10,
    TAPP_2_GEMM = 2,
    TAPP_3_GEMM = 3,
    TAPP_4_GEMM = 4,
    TAPP_5_GEMM = 5,
    TAPP_6_GEMM = 6,
    TAPP_7_GEMM = 7,
    TAPP_8_GEMM = 8,
    TAPP_9_GEMM = 9,
    TAPP_10_GEMM = 10,
    TAPP_11_GEMM = 11,
    TAPP_12_GEMM = 12,
    TAPP_13_GEMM = 13,
    TAPP_14_GEMM = 14,
    TAPP_15_GEMM = 15,
    TAPP_16_GEMM = 16,
    TAPP_17_GEMM = 17,
    TAPP_18_GEMM = 18,
    TAPP_19_GEMM = 19,
    TAPP_20_GEMM = 20,
};

#endif /* TAPP_DATATYPE_H_ */

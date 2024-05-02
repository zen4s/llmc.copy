#include "cuda_bf16.h"
#undef NDEBUG
#include "assert.h"
#include "float.h"
#include "stdio.h"

struct SplitFloatResult {
    nv_bfloat16 b_float;
    unsigned short bits;
};

template<class T, class S>
__host__ __device__ T bit_cast(S v) {
    T dest;
    static_assert(sizeof(v) == sizeof(dest));
    memcpy(&dest, &v, sizeof(v));
    return dest;
}

__host__  __device__ unsigned int float_as_uint(float f) {
    return bit_cast<unsigned int>(f);
}

__host__  __device__ unsigned short bfloat16_as_ushort(nv_bfloat16 f) {
    return bit_cast<unsigned short>(f);
}

__host__  __device__ float uint_as_float(unsigned int u) {
    return bit_cast<float>(u);
}

__host__  __device__ nv_bfloat16 ushort_as_bfloat16(unsigned short u) {
    return bit_cast<nv_bfloat16>(u);
}

// Splits a float into a bfloat16 and the remaining significant bits
__host__  __device__ SplitFloatResult split_float(float value, unsigned short threshold) {
    unsigned int float_bits = float_as_uint(value);
    // IEEE 754: float: S E(8) M (23)    bfloat: same, but significant 23-16 = 7 bits
    // ideally, we'd just store the cut-off 16 bits, but that doesn't work if rounding
    // is involved.
    unsigned int rounded_bits = float_bits & 0x0000FFFFu;
    if(rounded_bits > threshold) {
        SplitFloatResult result;
        result.b_float = __float2bfloat16_rn(uint_as_float(float_bits | 0xFFFFu));
        result.bits = rounded_bits & (~1u) | 1u;
        return result;
    } else {
        // truncation is easy
        SplitFloatResult result;
        result.b_float = ushort_as_bfloat16(float_bits >> 16u);
        result.bits = rounded_bits & (~1u);
        return result;
    }
}

// Reassembles a float from the bfloat16 part and the missing mantissa
__host__ __device__ float assemble_float(SplitFloatResult split) {
    constexpr const unsigned short BF16_SIGN_MASK        = 0b1'00000000'0000000u;
    constexpr const unsigned short BF16_EXPONENT_MASK    = 0b0'11111111'0000000u;
    constexpr const unsigned short BF16_SIGNIFICANT_MASK = 0b0'00000000'1111111u;
    unsigned short bf = bfloat16_as_ushort(split.b_float);
    if(split.bits & 1u) {
        // if we rounded away from zero, we need to undo these changes.
        // first, check if the significant (7 bits) of bf16 is zero
        if((bf & BF16_SIGNIFICANT_MASK) == 0) {
            // significant overflowed, need to decrement the exponent
            unsigned short exponent = (bf & BF16_EXPONENT_MASK) >> 7u;
            if(exponent == 0) {
                // zero, cannot be reached if we round away from zero
                __builtin_unreachable();
            }
            // decrement the exponent and set significant to all-ones
            bf = bf & BF16_SIGN_MASK | ((exponent-1) << 7u) | BF16_SIGNIFICANT_MASK;
        } else {
            // significant was incremented, decrement
            unsigned short significant = bf & BF16_SIGNIFICANT_MASK;
            bf = bf & (BF16_SIGN_MASK | BF16_EXPONENT_MASK) | (significant - 1);
        }
    }
    unsigned int result = (split.bits & (unsigned short)(~1u)) | (bf << 16u);
    return uint_as_float(result);
}


float round_trip(float f, unsigned short threshold) {
    SplitFloatResult split = split_float(f, threshold);
    float r = assemble_float(split);
    return r;
}

bool match_floats(float f1, float f2) {
    unsigned int u1 = float_as_uint(f1);
    unsigned int u2 = float_as_uint(f2);
    if((u1 & (~1u)) != (u2 & (~1u))) {
        printf("MISMATCH: %0b %0b\n", u1, u2);
        return false;
    }
    return true;
}

#define ASSERT_ROUND_TRIP(f) \
    assert(match_floats(f, round_trip(f, 0))); \
    assert(match_floats(f, round_trip(f, 0xFFFF)));  \

int main() {
    ASSERT_ROUND_TRIP(1.4623f)
    ASSERT_ROUND_TRIP(-63623.9f)
    ASSERT_ROUND_TRIP(FLT_TRUE_MIN)
    ASSERT_ROUND_TRIP(NAN)
    ASSERT_ROUND_TRIP(0)
    ASSERT_ROUND_TRIP(INFINITY)
    // make sure we trigger the "rounding increases exponent" code path
    float increment_exponent = bit_cast<float>((unsigned int)(0x40ff'fff0));
    ASSERT_ROUND_TRIP(increment_exponent)
    return EXIT_SUCCESS;
}
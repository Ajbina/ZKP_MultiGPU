#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------- CUDA error checking ----------------
#define CUDA_CALL(x) do { \
    cudaError_t _err = (x); \
    if (_err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// BN254 "toy" prime used in this project (2^64 - 2^32 + 1)
constexpr uint64_t BN254_PRIME = 0xFFFFFFFF00000001ULL;

// Strong inline (host+device)
#if defined(__CUDA_ARCH__)
  #define HD_INLINE __host__ __device__ __forceinline__
#else
  #define HD_INLINE __host__ __device__ inline
#endif

// ---------------- Field Element ----------------
struct fq {
    uint64_t value;

    HD_INLINE fq() : value(0) {}
    HD_INLINE explicit fq(uint64_t v) : value(v % BN254_PRIME) {}

    // 128-bit add then reduce mod p (handles uint64 overflow safely)
    HD_INLINE fq operator+(const fq& other) const {
        __uint128_t s = ( (__uint128_t)value + other.value );
        return fq( (uint64_t)(s % BN254_PRIME) );
    }

    // Subtract mod p
    HD_INLINE fq operator-(const fq& other) const {
        uint64_t diff = (value >= other.value) ? (value - other.value)
                                               : (BN254_PRIME + value - other.value);
        return fq(diff);
    }

    // 128-bit multiply then reduce mod p
    HD_INLINE fq operator*(const fq& other) const {
        __uint128_t prod = ( (__uint128_t)value ) * other.value;
        return fq((uint64_t)(prod % BN254_PRIME));
    }

    // Implemented in common.cu (host only)
    __host__ fq inverse() const;
    __host__ static fq pow(fq base, uint64_t exp);
};

// ---------------- Point type & helpers ----------------
struct ECPoint {
    fq X, Y, Z;  // Jacobian (X:Y:Z), ∞ iff Z == 0

    HD_INLINE ECPoint() : X(0), Y(1), Z(0) {}                    // ∞
    HD_INLINE ECPoint(fq x, fq y, fq z = fq(1)) : X(x), Y(y), Z(z) {}

    // Implemented in common.cu to mirror device formula (a=0 curve)
    __host__ __device__ ECPoint dbl() const;

    // 64-bit scalar multiply (double-and-add), implemented in common.cu
    __host__ __device__ ECPoint operator*(uint64_t scalar) const;

    // Unified add: routes to header‑inline Jacobian+Jacobian routine
    __host__ __device__ inline ECPoint operator+(const ECPoint& Q) const;

    // Host-only affine conversion (implemented in common.cu)
    __host__ void to_affine();
};

// Construct a point without running the full ctor
HD_INLINE ECPoint make_point(const fq& x, const fq& y, const fq& z) {
    ECPoint P; P.X = x; P.Y = y; P.Z = z; return P;
}

// ---------------- Header‑inline EC ops (fixes host undefined reference) ----------------

// Jacobian point doubling (a = 0 curve: y^2 = x^3 + 3)
//   A = X^2
//   B = Y^2
//   C = B^2
//   D = (X+B)^2 - A - C; D = 2D
//   E = 3A
//   F = E^2
//   X3 = F - 2D
//   Y3 = E*(D - X3) - 8*C
//   Z3 = 2*Y*Z
static HD_INLINE ECPoint ec_dbl_device(const ECPoint& P) {
    if (P.Z.value == 0) return P;  // ∞

    fq A  = P.X * P.X;
    fq B  = P.Y * P.Y;
    fq C  = B * B;
    fq D  = (P.X + B) * (P.X + B) - A - C;
    D     = D + D;                  // 2D
    fq E  = A + A + A;              // 3*X^2
    fq F  = E * E;
    fq X3 = F - D - D;              // F - 2D
    fq Y3 = E * (D - X3) - fq(8) * C; // *** -8*C ***
    fq Z3 = (P.Y * P.Z) + (P.Y * P.Z); // 2*Y*Z
    return make_point(X3, Y3, Z3);
}

// Jacobian point addition (a = 0), handles P==Q (doubling) and P==−P (∞)
static HD_INLINE ECPoint ec_add_jacobian_device(const ECPoint& P, const ECPoint& Q) {
    if (P.Z.value == 0) return Q;
    if (Q.Z.value == 0) return P;

    fq Z1Z1 = P.Z * P.Z;
    fq Z2Z2 = Q.Z * Q.Z;
    fq U1   = P.X * Z2Z2;            // X1 * Z2^2
    fq U2   = Q.X * Z1Z1;            // X2 * Z1^2
    fq S1   = P.Y * Q.Z * Z2Z2;      // Y1 * Z2^3
    fq S2   = Q.Y * P.Z * Z1Z1;      // Y2 * Z1^3

    if (U1.value == U2.value) {
        if (S1.value != S2.value) {
            // P + (-P) = ∞ (canonical ∞ = (0,1,0))
            return make_point(fq(0), fq(1), fq(0));
        } else {
            // P == Q -> doubling
            return ec_dbl_device(P);
        }
    }

    fq H  = U2 - U1;
    fq I  = fq(4) * H * H;           // (2H)^2
    fq J  = H * I;
    fq r  = fq(2) * (S2 - S1);
    fq V  = U1 * I;

    fq X3 = r * r - J - fq(2) * V;
    fq Y3 = r * (V - X3) - fq(2) * S1 * J;
    fq Z3 = fq(2) * P.Z * Q.Z * H;

    return make_point(X3, Y3, Z3);
}

// ECPoint::operator+ uses the header‑inline adder above
__host__ __device__ inline ECPoint ECPoint::operator+(const ECPoint& Q) const {
    return ec_add_jacobian_device(*this, Q);
}

// Scalar type used in this project
using scalar_t = uint64_t;

#undef HD_INLINE

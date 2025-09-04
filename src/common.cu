#include "common.cuh"

// -------------------- Field Arithmetic --------------------

__host__ fq fq::inverse() const {
    // Fermat's little theorem since p is prime
    return pow(*this, BN254_PRIME - 2);
}

__host__ fq fq::pow(fq base, uint64_t exp) {
    fq res(1);
    while (exp) {
        if (exp & 1) res = res * base;
        base = base * base;
        exp >>= 1;
    }
    return res;
}

// -------------------- Jacobian EC Point Arithmetic --------------------
// Use the same a=0 short‑Weierstrass formula as the device version.
// IMPORTANT: Y3 = E*(D - X3) - 8*C

__host__ __device__ ECPoint ECPoint::dbl() const {
    // Infinity check
    if (Z.value == 0) return *this;

    fq A = X * X;                    // X^2
    fq B = Y * Y;                    // Y^2
    fq C = B * B;                    // Y^4
    fq D = (X + B) * (X + B) - A - C; // (X+B)^2 - X^2 - Y^4
    D = D + D;                       // 2D
    fq E = A + A + A;                // 3X^2
    fq F = E * E;                    // E^2
    fq X3 = F - D - D;               // F - 2D
    fq Y3 = E * (D - X3) - fq(8) * C; // *** -8*C ***
    fq Z3 = (Y * Z) + (Y * Z);       // 2*Y*Z
    return make_point(X3, Y3, Z3);
}

// Simple left‑to‑right double‑and‑add on host (64‑bit scalars)
__host__ __device__ ECPoint ECPoint::operator*(uint64_t scalar) const {
    ECPoint R;          // ∞
    ECPoint Q = *this;
    for (int i = 0; i < 64; ++i) {
        if ((scalar >> i) & 1) R = R + Q;
        Q = Q.dbl();
    }
    return R;
}

// Convert Jacobian -> affine (host)
__host__ void ECPoint::to_affine() {
    if (Z.value == 0) return; // already infinity

    fq z_inv = Z.inverse();
    fq z2 = z_inv * z_inv;
    fq z3 = z2 * z_inv;

    X = X * z2;
    Y = Y * z3;
    Z = fq(1);
}

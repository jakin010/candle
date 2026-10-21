//! Branch-free f32 approximations of `erf` and `tanh`, and GELU built on them.
//!
//! Both are rational functions with no table lookups or data-dependent
//! branches, so a loop over a slice compiles to SIMD code (NEON, SSE/AVX,
//! SIMD128) without intrinsics. libm's `erff` and `tanhf` branch on the input
//! range and stay scalar.
//!
//! They back the opt-in fast GELU path, see [`crate::utils::set_fast_gelu`].
#![allow(clippy::excessive_precision)]

/// `a * b + c`, fused where the target has a fused multiply-add instruction.
///
/// Without one (x86-64 builds without the `fma` target feature) `mul_add`
/// becomes a call to libm's `fmaf` per element, which stops vectorization and
/// is slower than the scalar code these functions replace.
#[inline(always)]
fn madd(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        a * b + c
    }
}

/// `erf(x)` for f32, as a degree 13/8 rational approximation, and exactly ±1
/// from `|x| = 3.919206` on, where `erf` rounds to ±1 in f32.
///
/// The coefficients are the ones Eigen (`generic_fast_erf_float`) and XLA use.
/// The absolute error is below 1e-6 over the whole range; libm's `erff` is
/// within about 1e-7. NaN propagates.
#[inline(always)]
pub fn erf_f32(x: f32) -> f32 {
    let input = x;
    let x = x.clamp(-4.0, 4.0);
    let x2 = x * x;
    // Odd numerator, x * P(x^2).
    let mut p = madd(x2, -2.726_142_3e-10, 2.770_681_4e-8);
    p = madd(x2, p, -2.101_024e-6);
    p = madd(x2, p, -5.692_506_4e-5);
    p = madd(x2, p, -7.349_906_3e-4);
    p = madd(x2, p, -2.954_599_8e-3);
    p = madd(x2, p, -1.609_603_3e-2);
    // Even denominator, Q(x^2).
    let mut q = madd(x2, -1.456_607_2e-5, -2.133_740_6e-4);
    q = madd(x2, q, -1.682_827e-3);
    q = madd(x2, q, -7.373_329e-3);
    q = madd(x2, q, -1.426_473_9e-2);
    let y = x * p / q;
    // A select rather than a branch, so loops still vectorize.
    if input.abs() >= 3.919_206 {
        1f32.copysign(input)
    } else {
        y
    }
}

/// `tanh(x)` for f32, as a degree 13/6 rational approximation on
/// `[-7.905, 7.905]` (held at its end values up to `|x| = 9.010913`, where
/// `tanh` rounds to ±1 in f32, and exactly ±1 beyond), and `tanh(x) = x` for
/// `|x| < 0.0004`.
///
/// The coefficients are the ones Eigen (`generic_fast_tanh_float`) uses. The
/// absolute error is below 1e-6 over the whole range. NaN propagates.
#[inline(always)]
pub fn tanh_f32(x: f32) -> f32 {
    let clamped = x.clamp(-7.905_311, 7.905_311);
    let x2 = clamped * clamped;
    // Odd numerator, x * P(x^2).
    let mut p = madd(x2, -2.760_768_5e-16, 2.000_187_9e-13);
    p = madd(x2, p, -8.604_671_5e-11);
    p = madd(x2, p, 5.122_297e-8);
    p = madd(x2, p, 1.485_722_4e-5);
    p = madd(x2, p, 6.372_619_3e-4);
    p = madd(x2, p, 4.893_524_6e-3);
    // Even denominator, Q(x^2).
    let mut q = madd(x2, 1.198_258_4e-6, 1.185_347_1e-4);
    q = madd(x2, q, 2.268_434_6e-3);
    q = madd(x2, q, 4.893_525e-3);
    let y = clamped * p / q;
    // Selects rather than branches, so loops still vectorize.
    if x.abs() < 0.0004 {
        x
    } else if x.abs() >= 9.010_913 {
        1f32.copysign(x)
    } else {
        y
    }
}

/// GELU, `0.5 * x * (1 + erf(x / sqrt(2)))`, with [`erf_f32`].
#[inline(always)]
pub fn gelu_erf(x: f32) -> f32 {
    (erf_f32(x * std::f32::consts::FRAC_1_SQRT_2) + 1.) * 0.5 * x
}

/// The tanh approximation of GELU, with [`tanh_f32`].
#[inline(always)]
pub fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + tanh_f32(crate::op::SQRT_TWO_OVER_PI_F32 * x * (1.0 + 0.044715 * x * x)))
}

macro_rules! slice_fn {
    ($(#[$doc:meta])* $name:ident, $f:ident) => {
        $(#[$doc])*
        pub fn $name(xs: &[f32], ys: &mut [f32]) {
            #[inline(always)]
            fn run(xs: &[f32], ys: &mut [f32]) {
                for (y, &x) in ys.iter_mut().zip(xs) {
                    *y = $f(x)
                }
            }
            // Baseline x86-64 builds only get 4-wide SSE2 vectors; compile the
            // loop for AVX2 too and pick it at run time. The arithmetic is the
            // same (no fused ops are added), so results do not change.
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2")]
                unsafe fn run_avx2(xs: &[f32], ys: &mut [f32]) {
                    run(xs, ys)
                }
                if std::arch::is_x86_feature_detected!("avx2") {
                    // SAFETY: the CPU supports AVX2.
                    return unsafe { run_avx2(xs, ys) };
                }
            }
            run(xs, ys)
        }
    };
}

slice_fn!(
    /// `ys[i] = gelu_erf(xs[i])`.
    gelu_erf_vec,
    gelu_erf
);
slice_fn!(
    /// `ys[i] = gelu_tanh(xs[i])`.
    gelu_tanh_vec,
    gelu_tanh
);

#[cfg(test)]
mod tests {
    use super::*;

    /// Evenly spaced points over `[lo, hi]`, both ends included.
    fn sweep(lo: f32, hi: f32, n: usize) -> impl Iterator<Item = f32> {
        (0..n).map(move |i| lo + (hi - lo) * i as f32 / (n - 1) as f32)
    }

    fn max_abs_error(f: impl Fn(f32) -> f32, exact: impl Fn(f64) -> f64) -> (f64, f32) {
        sweep(-10.0, 10.0, 2_000_001)
            .map(|x| ((f(x) as f64 - exact(x as f64)).abs(), x))
            .fold((0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a })
    }

    #[test]
    fn erf_is_within_1e6_of_exact() {
        let (err, at) = max_abs_error(erf_f32, libm::erf);
        assert!(err < 1e-6, "max |erf_f32 - erf| = {err:e} at x = {at}");
    }

    #[test]
    fn tanh_is_within_1e6_of_exact() {
        let (err, at) = max_abs_error(tanh_f32, f64::tanh);
        assert!(err < 1e-6, "max |tanh_f32 - tanh| = {err:e} at x = {at}");
    }

    #[test]
    fn special_values() {
        for f in [erf_f32, tanh_f32] {
            assert_eq!(f(0.0), 0.0);
            assert_eq!(f(f32::INFINITY), 1.0);
            assert_eq!(f(f32::NEG_INFINITY), -1.0);
            assert_eq!(f(1e6), 1.0);
            assert_eq!(f(-1e6), -1.0);
            assert!(f(f32::NAN).is_nan());
        }
        // Saturated exactly where the f32 functions round to ±1.
        assert_eq!(erf_f32(3.92), 1.0);
        assert_eq!(tanh_f32(-9.02), -1.0);
        // Odd functions, exactly.
        for x in sweep(0.0, 10.0, 10_001) {
            assert_eq!(erf_f32(-x), -erf_f32(x));
            assert_eq!(tanh_f32(-x), -tanh_f32(x));
        }
        // Tiny inputs keep their relative precision.
        assert_eq!(tanh_f32(1e-30), 1e-30);
        assert!((erf_f32(1e-30) / 1e-30 - std::f32::consts::FRAC_2_SQRT_PI).abs() < 1e-6);
    }
}

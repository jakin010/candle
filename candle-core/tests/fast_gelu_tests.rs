//! The fast GELU path (feature `fast-gelu`, switched by
//! `candle_core::utils::set_fast_gelu`).
//!
//! The switch is process-wide, so these checks live in their own test binary
//! and in a single test, where nothing else can flip it concurrently.
#![cfg(feature = "fast-gelu")]
use candle_core::{cpu, utils, Device, Result, Tensor};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
fn fast_gelu_is_close_to_exact_and_can_be_turned_off() -> Result<()> {
    // Large enough for the parallel element-wise path; a dense sweep of [-10, 10].
    let n = 1 << 18;
    let xs: Vec<f32> = (0..n)
        .map(|i| -10.0 + 20.0 * i as f32 / (n - 1) as f32)
        .collect();
    let x = Tensor::from_vec(xs.clone(), n, &Device::Cpu)?;
    // The same values, laid out so each element goes through the scalar path.
    let x_strided = Tensor::from_vec(xs.clone(), (2, n / 2), &Device::Cpu)?.t()?;
    let unstride = |t: Tensor| -> Result<Vec<f32>> { t.t()?.flatten_all()?.to_vec1() };

    if std::env::var_os("CANDLE_FAST_GELU").is_none() {
        assert!(
            utils::fast_gelu(),
            "with the feature, the fast path is on by default"
        );
    }

    // Off: bit-identical to libm, contiguous or not.
    utils::set_fast_gelu(false);
    let erf_exact = x.gelu_erf()?.to_vec1::<f32>()?;
    let tanh_exact = x.gelu()?.to_vec1::<f32>()?;
    let libm_erf: Vec<f32> = xs
        .iter()
        .map(|&v| (cpu::erf::erf_f32(v * std::f32::consts::FRAC_1_SQRT_2) + 1.) * 0.5 * v)
        .collect();
    assert_eq!(erf_exact, libm_erf);
    assert_eq!(unstride(x_strided.gelu_erf()?)?, erf_exact);
    let vendor_gelu = utils::has_mkl() || utils::has_accelerate();
    if !vendor_gelu {
        // With mkl/accelerate, contiguous gelu uses their tanh and strided
        // gelu libm's, so the two differ in the last bits either way.
        assert_eq!(unstride(x_strided.gelu()?)?, tanh_exact);
    }

    // On: close to the exact values, and the same for every layout.
    utils::set_fast_gelu(true);
    let erf_fast = x.gelu_erf()?.to_vec1::<f32>()?;
    let tanh_fast = x.gelu()?.to_vec1::<f32>()?;
    let erf_err = max_abs_diff(&erf_fast, &erf_exact);
    assert!(erf_err < 2e-6, "gelu_erf: max |fast - exact| = {erf_err:e}");
    assert_ne!(erf_fast, erf_exact, "gelu_erf did not take the fast path");
    assert_eq!(unstride(x_strided.gelu_erf()?)?, erf_fast);
    if vendor_gelu {
        // Their own vectorized gelu is used either way.
        assert_eq!(tanh_fast, tanh_exact);
    } else {
        let tanh_err = max_abs_diff(&tanh_fast, &tanh_exact);
        assert!(tanh_err < 2e-6, "gelu: max |fast - exact| = {tanh_err:e}");
        assert_ne!(tanh_fast, tanh_exact, "gelu did not take the fast path");
        assert_eq!(unstride(x_strided.gelu()?)?, tanh_fast);
    }

    // And off again.
    utils::set_fast_gelu(false);
    assert_eq!(x.gelu_erf()?.to_vec1::<f32>()?, erf_exact);
    assert_eq!(x.gelu()?.to_vec1::<f32>()?, tanh_exact);
    Ok(())
}

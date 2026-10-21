//! Ops that the CPU backend runs in parallel on large inputs (64K elements and
//! up) must give exactly what the serial path gives.
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};

const ROWS: usize = 300;
const COLS: usize = 512; // ROWS * COLS is above the parallel threshold, one row is not.

fn input() -> Result<(Tensor, Vec<f32>)> {
    let t = Tensor::randn(0f32, 3., (ROWS, COLS), &Device::Cpu)?;
    let v = t.flatten_all()?.to_vec1::<f32>()?;
    Ok((t, v))
}

#[test]
fn elementwise_maps_match_scalar_code() -> Result<()> {
    let (x, xs) = input()?;
    let ge: Vec<u8> = xs.iter().map(|&v| u8::from(v >= 0.5)).collect();
    assert_eq!(x.ge(0.5f32)?.flatten_all()?.to_vec1::<u8>()?, ge);
    let y = x.affine(0.125, -1.5)?;
    let affine: Vec<f32> = xs.iter().map(|&v| v * 0.125f32 + -1.5f32).collect();
    assert_eq!(y.flatten_all()?.to_vec1::<f32>()?, affine);
    let f16: Vec<half::f16> = xs.iter().map(|&v| half::f16::from_f32(v)).collect();
    assert_eq!(
        x.to_dtype(DType::F16)?
            .flatten_all()?
            .to_vec1::<half::f16>()?,
        f16
    );
    let ids: Vec<i64> = (0..(ROWS * COLS) as i64)
        .map(|i| i * 7 % 1000 - 500)
        .collect();
    let ids_f32: Vec<f32> = ids.iter().map(|&v| v as f32).collect();
    let ids = Tensor::from_vec(ids, (ROWS, COLS), &Device::Cpu)?;
    assert_eq!(
        ids.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
        ids_f32
    );
    Ok(())
}

#[test]
fn gather_matches_indexing() -> Result<()> {
    let (x, xs) = input()?;
    // Along the last dim: every row reverses its columns.
    let ids: Vec<u32> = (0..ROWS * COLS)
        .map(|i| (COLS - 1 - i % COLS) as u32)
        .collect();
    let ids = Tensor::from_vec(ids, (ROWS, COLS), &Device::Cpu)?;
    let expected: Vec<f32> = (0..ROWS * COLS)
        .map(|i| xs[i / COLS * COLS + COLS - 1 - i % COLS])
        .collect();
    assert_eq!(
        x.gather(&ids, D::Minus1)?.flatten_all()?.to_vec1::<f32>()?,
        expected
    );
    // Along the first dim: column j reads row (r + j) % ROWS.
    let ids: Vec<u32> = (0..ROWS * COLS)
        .map(|i| ((i / COLS + i % COLS) % ROWS) as u32)
        .collect();
    let ids = Tensor::from_vec(ids, (ROWS, COLS), &Device::Cpu)?;
    let expected: Vec<f32> = (0..ROWS * COLS)
        .map(|i| xs[(i / COLS + i % COLS) % ROWS * COLS + i % COLS])
        .collect();
    assert_eq!(
        x.gather(&ids, 0)?.flatten_all()?.to_vec1::<f32>()?,
        expected
    );
    Ok(())
}

#[test]
fn last_dim_reductions_match_row_by_row() -> Result<()> {
    let (x, _) = input()?;
    let max = x.max_keepdim(D::Minus1)?.flatten_all()?.to_vec1::<f32>()?;
    let argmax = x
        .argmax_keepdim(D::Minus1)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    let sum = x.sum_keepdim(D::Minus1)?.flatten_all()?.to_vec1::<f32>()?;
    for r in 0..ROWS {
        // A single row is below the threshold, so it takes the serial path.
        let row = x.i(r)?;
        assert_eq!(max[r], row.max(0)?.to_scalar::<f32>()?, "max, row {r}");
        assert_eq!(
            argmax[r],
            row.argmax(0)?.to_scalar::<u32>()?,
            "argmax, row {r}"
        );
        assert_eq!(sum[r], row.sum(0)?.to_scalar::<f32>()?, "sum, row {r}");
    }
    Ok(())
}

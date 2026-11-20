//! Evaluate each side of a one-dimensional lane mask on its own lanes only.

use super::error::{Arithmetic, HydraulicResult};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::CertifiedExpansion;

/// The lanes one side of a split evaluates.
pub(crate) enum Lanes<R: Runtime<DType = DType>> {
    /// Every lane of the mask, in order.
    All,
    /// Ascending lane indices along dimension zero.
    Rows(Tensor<R>),
}

impl<R> Lanes<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Return the nonzero lanes of a one-dimensional mask, or `None` when it has none.
    pub(crate) fn of(c: &R::Client, mask: &Tensor<R>) -> HydraulicResult<Option<Self>> {
        let rows = c.nonzero(mask)?.flatten()?;
        Ok(if rows.numel() == 0 {
            None
        } else if rows.numel() == mask.numel() {
            Some(Self::All)
        } else {
            Some(Self::Rows(rows))
        })
    }

    /// Select `value` on these lanes along dimension zero.
    pub(crate) fn tensor(&self, c: &R::Client, value: &Tensor<R>) -> HydraulicResult<Tensor<R>> {
        Ok(match self {
            Self::All => value.clone(),
            Self::Rows(rows) => c.index_select(value, 0, rows)?,
        })
    }

    /// Write lane results into a copy of `fill`. Other lanes keep `fill`.
    pub(crate) fn place(
        &self,
        c: &R::Client,
        fill: &Tensor<R>,
        value: &Tensor<R>,
    ) -> HydraulicResult<Tensor<R>> {
        Ok(match self {
            Self::All => value.clone(),
            Self::Rows(rows) => c.index_put(fill, 0, rows, value)?,
        })
    }

    /// Capture `value` on these lanes with the normalization of a masked capture.
    pub(crate) fn expansion(
        &self,
        a: &mut Arithmetic<'_, R>,
        value: &CertifiedExpansion<R>,
    ) -> HydraulicResult<CertifiedExpansion<R>> {
        let c = a.client();
        Ok(a.expansion_bounds(
            &self.tensor(c, value.high())?,
            &self.tensor(c, value.low())?,
            &self.tensor(c, value.abs_error())?,
        )?)
    }
}

/// Select `yes` on nonzero mask lanes and `no` on zero mask lanes.
///
/// Each side runs only on its own lanes, and a side with no lanes never runs.
/// Lane results match `choose_expansion` over both sides evaluated on every lane
/// when each side is lane-wise and its inactive lanes would hold benign operands.
pub(crate) fn split_expansion<R, Y, N>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    yes: Y,
    no: N,
) -> HydraulicResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
    Y: FnOnce(&mut Arithmetic<'_, R>, &Lanes<R>) -> HydraulicResult<CertifiedExpansion<R>>,
    N: FnOnce(&mut Arithmetic<'_, R>, &Lanes<R>) -> HydraulicResult<CertifiedExpansion<R>>,
{
    let c = a.client();
    let on_lanes = match Lanes::of(c, mask)? {
        Some(Lanes::All) => {
            let on = yes(a, &Lanes::All)?;
            return Ok(a.expansion_bounds(on.high(), on.low(), on.abs_error())?);
        }
        None => {
            let off = no(a, &Lanes::All)?;
            return Ok(a.expansion_bounds(off.high(), off.low(), off.abs_error())?);
        }
        Some(lanes) => lanes,
    };
    // Both sides hold lanes here.
    let zero = Tensor::zeros(mask.shape(), mask.dtype(), c.device())?;
    let off_lanes = Lanes::Rows(c.nonzero(&c.eq(mask, &zero)?)?.flatten()?);
    let on = yes(a, &on_lanes)?;
    let off = no(a, &off_lanes)?;
    let dense = |on: &Tensor<R>, off: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        off_lanes.place(c, &on_lanes.place(c, &zero, on)?, off)
    };
    Ok(a.expansion_bounds(
        &dense(on.high(), off.high())?,
        &dense(on.low(), off.low())?,
        &dense(on.abs_error(), off.abs_error())?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::super::equation_ops::choose_expansion;
    use super::super::error::{Arithmetic, HydraulicResult, checked};
    use super::{Lanes, split_expansion};
    use numr::runtime::RuntimeClient;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;
    use solvr::common::arithmetic::CertifiedExpansion;
    use std::cell::Cell;

    type Expansion = CertifiedExpansion<CpuRuntime>;

    fn bits(value: &Expansion) -> Vec<[u64; 3]> {
        let parts = [value.high(), value.low(), value.abs_error()].map(|t| t.to_vec::<f64>());
        (0..parts[0].len())
            .map(|i| parts.each_ref().map(|p| p[i].to_bits()))
            .collect()
    }

    fn lanes(a: &mut Arithmetic<'_, CpuRuntime>, values: &[f64]) -> HydraulicResult<Expansion> {
        let device = a.client().device().clone();
        let center = Tensor::<CpuRuntime>::from_slice(values, &[values.len()], &device)?;
        let third =
            Tensor::<CpuRuntime>::from_slice(&vec![3.0; values.len()], &[values.len()], &device)?;
        let point = a.expansion_point(&center)?;
        let divisor = a.expansion_point(&third)?;
        Ok(a.expansion_div_positive(&point, &divisor)?)
    }

    /// Split evaluation matches eager evaluation and runs each side only on its lanes.
    fn compare(mask: &[f64]) -> HydraulicResult<()> {
        let client = CpuClient::new(CpuDevice::new());
        checked::<CpuRuntime, _>(&client, "lane split test", |a| {
            let n = mask.len();
            let x = lanes(a, &(0..n).map(|i| 1.0 + i as f64).collect::<Vec<_>>())?;
            let y = lanes(a, &(0..n).map(|i| 7.0 - i as f64 * 0.5).collect::<Vec<_>>())?;
            let device = a.client().device().clone();
            let mask = Tensor::<CpuRuntime>::from_slice(mask, &[n], &device)?;
            let eager_yes = a.expansion_mul(&x, &y)?;
            let eager_no = a.expansion_div_positive(&x, &y)?;
            let eager = choose_expansion(a, &mask, &eager_yes, &eager_no)?;
            let live = mask.to_vec::<f64>().iter().filter(|&&m| m != 0.0).count();
            let (yes_lanes, no_lanes) = (Cell::new(None), Cell::new(None));
            let split = split_expansion(
                a,
                &mask,
                |a, lanes: &Lanes<CpuRuntime>| {
                    let (x, y) = (lanes.expansion(a, &x)?, lanes.expansion(a, &y)?);
                    yes_lanes.set(Some(x.high().numel()));
                    Ok(a.expansion_mul(&x, &y)?)
                },
                |a, lanes: &Lanes<CpuRuntime>| {
                    let (x, y) = (lanes.expansion(a, &x)?, lanes.expansion(a, &y)?);
                    no_lanes.set(Some(x.high().numel()));
                    Ok(a.expansion_div_positive(&x, &y)?)
                },
            )?;
            assert_eq!(bits(&split), bits(&eager));
            assert_eq!(yes_lanes.get(), (live != 0).then_some(live));
            assert_eq!(no_lanes.get(), (live != n || n == 0).then_some(n - live));
            Ok(())
        })
    }

    #[test]
    fn mixed_masks_match_eager_selection() -> HydraulicResult<()> {
        compare(&[1.0, 0.0, 0.0, 1.0, 0.0])?;
        compare(&[0.0, 1.0])
    }

    #[test]
    fn uniform_masks_skip_the_empty_side() -> HydraulicResult<()> {
        compare(&[1.0, 1.0, 1.0])?;
        compare(&[0.0, 0.0, 0.0])
    }

    #[test]
    fn empty_masks_run_one_empty_side() -> HydraulicResult<()> {
        compare(&[])
    }

    #[test]
    fn lanes_select_and_place_mask_rows() -> HydraulicResult<()> {
        let client = CpuClient::new(CpuDevice::new());
        let device = client.device().clone();
        let tensor =
            |values: &[f64]| Tensor::<CpuRuntime>::from_slice(values, &[values.len()], &device);
        assert!(Lanes::of(&client, &tensor(&[0.0, 0.0])?)?.is_none());
        assert!(matches!(
            Lanes::of(&client, &tensor(&[1.0, 1.0])?)?,
            Some(Lanes::All)
        ));
        let lanes = Lanes::of(&client, &tensor(&[0.0, 1.0, 0.0, 1.0])?)?;
        assert!(matches!(lanes, Some(Lanes::Rows(_))));
        let Some(lanes) = lanes else {
            return Ok(());
        };
        let value = tensor(&[10.0, 11.0, 12.0, 13.0])?;
        let selected = lanes.tensor(&client, &value)?;
        assert_eq!(selected.to_vec::<f64>(), [11.0, 13.0]);
        let placed = lanes.place(&client, &tensor(&[-1.0; 4])?, &tensor(&[5.0, 6.0])?)?;
        assert_eq!(placed.to_vec::<f64>(), [-1.0, 5.0, -1.0, 6.0]);
        let all = Lanes::<CpuRuntime>::All;
        assert_eq!(
            all.tensor(&client, &value)?.to_vec::<f64>(),
            value.to_vec::<f64>()
        );
        assert_eq!(
            all.place(&client, &tensor(&[0.0; 4])?, &value)?
                .to_vec::<f64>(),
            value.to_vec::<f64>()
        );
        Ok(())
    }
}

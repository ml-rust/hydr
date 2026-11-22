//! Per-row operands of the arc layout, with a harmless admitted barrel on other rows.

use crate::structures::{CulvertBarrel, CulvertBarrelShape};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Span, rise, corner and bottom radius of an admitted pipe-arch.
const HARMLESS_PIPE_ARCH: [f64; 4] = [2.0, 1.5, 0.5, 2.0];
/// Span, rise and side radius of an admitted horizontal ellipse.
const HARMLESS_HORIZONTAL: [f64; 3] = [2.0, 1.0, 0.25];
/// Span, rise and end radius of an admitted vertical ellipse.
const HARMLESS_VERTICAL: [f64; 3] = [1.0, 2.0, 0.25];
/// Span and rise of an admitted flat-floor arch.
const HARMLESS_ARCH: [f64; 2] = [2.0, 1.0];

/// `[m]` row masks and operand columns. Each family's columns hold the barrel's own dimensions
/// on its rows and a harmless admitted barrel elsewhere, so every family runs on every row.
pub(crate) struct ArcInputs<R: Runtime<DType = DType>> {
    /// 1 on pipe-arch rows.
    pub(crate) pipe_arch: Tensor<R>,
    /// 1 on horizontal-ellipse rows.
    pub(crate) horizontal: Tensor<R>,
    /// 1 on flat-floor arch rows.
    pub(crate) arch: Tensor<R>,
    /// Span, rise, corner radius and bottom radius.
    pub(crate) pipe_arch_operands: [Tensor<R>; 4],
    /// Span, rise and side radius.
    pub(crate) horizontal_operands: [Tensor<R>; 3],
    /// Span, rise and end radius.
    pub(crate) vertical_operands: [Tensor<R>; 3],
    /// Span and rise.
    pub(crate) arch_operands: [Tensor<R>; 2],
}

impl<R: Runtime<DType = DType>> ArcInputs<R> {
    /// Uploads the masks and operand columns of `barrels`.
    pub(crate) fn new(barrels: &[CulvertBarrel], device: &R::Device) -> numr::error::Result<Self> {
        let m = barrels.len();
        let column = |f: &dyn Fn(&CulvertBarrel) -> f64| -> numr::error::Result<Tensor<R>> {
            let values: Vec<f64> = barrels.iter().map(f).collect();
            Tensor::<R>::from_slice(&values, &[m], device)
        };
        let mask = |shape: CulvertBarrelShape| column(&|b| f64::from(u8::from(b.shape() == shape)));
        // Operand `k` of `shape` rows: span, rise, then the given radii.
        let operand = |shape: CulvertBarrelShape, k: usize, harmless: f64| {
            column(&|b| {
                let given = b.given_radius_m().unwrap_or(&[]);
                let own = [b.span_m(), b.rise_m()]
                    .into_iter()
                    .chain(given.iter().copied())
                    .nth(k);
                match own {
                    Some(value) if b.shape() == shape => value,
                    _ => harmless,
                }
            })
        };
        use CulvertBarrelShape::{Arch, HorizontalEllipse, PipeArch, VerticalEllipse};
        let [pb, pd, pc, pr] = HARMLESS_PIPE_ARCH;
        let [hb, hd, hr] = HARMLESS_HORIZONTAL;
        let [vb, vd, vr] = HARMLESS_VERTICAL;
        let [ab, ad] = HARMLESS_ARCH;
        Ok(Self {
            pipe_arch: mask(PipeArch)?,
            horizontal: mask(HorizontalEllipse)?,
            arch: mask(Arch)?,
            pipe_arch_operands: [
                operand(PipeArch, 0, pb)?,
                operand(PipeArch, 1, pd)?,
                operand(PipeArch, 2, pc)?,
                operand(PipeArch, 3, pr)?,
            ],
            horizontal_operands: [
                operand(HorizontalEllipse, 0, hb)?,
                operand(HorizontalEllipse, 1, hd)?,
                operand(HorizontalEllipse, 2, hr)?,
            ],
            vertical_operands: [
                operand(VerticalEllipse, 0, vb)?,
                operand(VerticalEllipse, 1, vd)?,
                operand(VerticalEllipse, 2, vr)?,
            ],
            arch_operands: [operand(Arch, 0, ab)?, operand(Arch, 1, ad)?],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::StructureError;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn family_columns_hold_own_dimensions_or_the_harmless_barrel() -> Result<(), StructureError> {
        let device = CpuDevice::new();
        let barrels = [
            CulvertBarrel::circular(0.3)?,
            CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?,
            CulvertBarrel::horizontal_ellipse(1.524, 0.9652, 0.419_475_92)?,
            CulvertBarrel::arch(3.048, 1.6002)?,
        ];
        let inputs = ArcInputs::<CpuRuntime>::new(&barrels, &device)?;
        assert_eq!(inputs.pipe_arch.try_to_vec::<f64>()?, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(inputs.horizontal.try_to_vec::<f64>()?, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(inputs.arch.try_to_vec::<f64>()?, [0.0, 0.0, 0.0, 1.0]);
        let [_, _, _, bottom] = &inputs.pipe_arch_operands;
        assert_eq!(bottom.try_to_vec::<f64>()?, [2.0, 1.938_528, 2.0, 2.0]);
        let [_, _, side] = &inputs.horizontal_operands;
        assert_eq!(side.try_to_vec::<f64>()?, [0.25, 0.25, 0.419_475_92, 0.25]);
        let [span, rise] = &inputs.arch_operands;
        assert_eq!(span.try_to_vec::<f64>()?, [2.0, 2.0, 2.0, 3.048]);
        assert_eq!(rise.try_to_vec::<f64>()?, [1.0, 1.0, 1.0, 1.6002]);
        let [_, _, end] = &inputs.vertical_operands;
        assert_eq!(end.try_to_vec::<f64>()?, [0.25; 4]);
        // Each harmless barrel is itself admitted.
        let [b, d, c, r] = HARMLESS_PIPE_ARCH;
        CulvertBarrel::pipe_arch(b, d, c, r)?;
        let [b, d, r] = HARMLESS_HORIZONTAL;
        CulvertBarrel::horizontal_ellipse(b, d, r)?;
        let [b, d, r] = HARMLESS_VERTICAL;
        CulvertBarrel::vertical_ellipse(b, d, r)?;
        let [b, d] = HARMLESS_ARCH;
        CulvertBarrel::arch(b, d)?;
        Ok(())
    }
}

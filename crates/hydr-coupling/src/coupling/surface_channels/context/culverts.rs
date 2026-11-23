//! Invert stops of the culvert links: the static operands of the invert cap.

use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::structures::LinkLaw;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Culvert links with the stage and plan area at which each end stops giving water.
///
/// A culvert donor stops at its invert stop, as a pump donor stops at its suction stop. Other
/// links hold zeros.
#[derive(Clone)]
pub(in super::super) struct CulvertStops<R: Runtime<DType = DType>> {
    /// One at every culvert link, `[L]`.
    pub mask: Tensor<R>,
    /// Invert stop of endpoints A and B, each `[L]`.
    pub stop: [Tensor<R>; 2],
    /// Plan area of endpoints A and B at their invert stops, each `[L]`.
    pub area: [Tensor<R>; 2],
    /// Every link is a culvert.
    pub every: bool,
}

impl<R: Runtime<DType = DType>> CulvertStops<R> {
    /// Stops of every culvert among `laws`, or `None` without a culvert.
    ///
    /// `side_area(link, side, stage)` is the plan area of one link end at `stage`.
    pub fn new(
        laws: &[LinkLaw],
        side_area: impl Fn(usize, usize, f64) -> f64,
        device: &R::Device,
    ) -> Result<Option<Self>, SurfaceChannelError> {
        if laws.iter().all(|law| law.culvert().is_none()) {
            return Ok(None);
        }
        let column =
            |value: &dyn Fn(usize, &LinkLaw) -> f64| -> Result<Tensor<R>, SurfaceChannelError> {
                let values: Vec<f64> = laws
                    .iter()
                    .enumerate()
                    .map(|(link, law)| value(link, law))
                    .collect();
                Ok(Tensor::<R>::from_slice(&values, &[values.len()], device)?)
            };
        let stop = |side: usize| column(&|_, law| law.culvert().map_or(0.0, |k| k.stop_m[side]));
        let area = |side: usize| {
            column(&|link, law| {
                law.culvert()
                    .map_or(0.0, |k| side_area(link, side, k.stop_m[side]))
            })
        };
        Ok(Some(Self {
            mask: column(&|_, law| f64::from(u8::from(law.culvert().is_some())))?,
            stop: [stop(0)?, stop(1)?],
            area: [area(0)?, area(1)?],
            every: laws.iter().all(|law| law.culvert().is_some()),
        }))
    }
}

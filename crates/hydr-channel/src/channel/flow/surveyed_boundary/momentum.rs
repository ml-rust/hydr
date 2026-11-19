//! The `Discharge` trace momentum flux through its exact discharge identity.
//!
//! The exact trace satisfies `A u_b = q`, so its momentum flux is `F(d) = q² / A + g I`.
//! With `dI/dd = A` and `dA/dd = T`, `dF/dd = T (c² - q² / A²)`. Where `A c > |q|` holds over
//! the depth enclosure, `F` increases there, and `F` at the two end depths encloses the
//! momentum flux. Its width follows `T (c - u)(c + u)` times the depth width. Direct interval
//! evaluation of `A u² + g I` over the same enclosure widens with `T (c + u)²`, and a root near
//! the sonic depth has a depth width that grows like `1 / (c - u)`.

use super::{client::TraceClient, error::trace_error, interior::Interior, point::TracePoint};
use crate::channel::flow::{
    ChannelError,
    context::Context,
    surveyed::{
        equation_ops::scalar,
        error::{HydraulicResult, checked},
    },
};
use numr::{dtype::DType, ops::CompareOps, runtime::Runtime, tensor::Tensor};
use solvr::common::arithmetic::Interval;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TraceClient<R>,
{
    /// Momentum flux enclosure of a `Discharge` trace from `F(d) = q² / A + g I`, or `None`.
    ///
    /// `point` encloses the trace over its certified root depth enclosure. `target` is the
    /// outward discharge `q`. Returns `None` unless `A c > |q|` is certified over `point`.
    pub(super) fn surveyed_discharge_momentum(
        &self,
        interior: &Interior<R>,
        point: &TracePoint<R>,
        target: &Tensor<R>,
    ) -> Result<Option<Interval<R>>, ChannelError> {
        let sources = &interior.sources;
        let c = sources.geometry.client();
        let gravity = self.definition.config.gravity;
        let (increasing, flux) = checked(c, "surveyed discharge momentum identity", |a| {
            let q = a.point(target)?;
            let squared = a.mul(&q, &q)?;
            // Over the enclosure A >= A.lower and c >= c.lower, so A c exceeds their product.
            let area = a.point(point.area.lower())?;
            let sound = a.point(point.sound.lower())?;
            let capacity = a.mul(&area, &sound)?;
            let capacity = a.mul(&capacity, &capacity)?;
            let increasing = c.gt(capacity.lower(), squared.upper())?;
            let g = scalar(a, gravity)?;
            let mut at = |depth: &Tensor<R>| -> HydraulicResult<Interval<R>> {
                let depth = a.point(depth)?;
                let state = sources.geometry.at_depth(a, &sources.bands, &depth, None)?;
                let advective = a.div_positive_interval(&squared, &state.area)?;
                let pressure = a.mul(&g, &state.integral)?;
                Ok(a.add(&advective, &pressure)?)
            };
            let shallow = at(point.depth.lower())?;
            let deep = at(point.depth.upper())?;
            Ok((increasing, a.bounds(shallow.lower(), deep.upper())?))
        })
        .map_err(trace_error)?;
        Ok(self.all(&increasing)?.then_some(flux))
    }
}

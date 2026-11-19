//! Sonic outflow capacity margins of surveyed `Discharge` ends.
//!
//! A leaving trace lies on the rarefaction side curve of the end cell. Along that curve the
//! outward discharge `A u_b` peaks at the sonic depth `d_s`, where `u_b = c`. A target above
//! that peak has no subcritical root. The margin `A u_b (d_s) - q` measures how much outflow
//! the end cell has left. The coordinator reads it to tell a capacity loss from other trace
//! rejections.

use super::{client::TraceClient, error::trace_error, point::Evaluation, request::TraceRequest};
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary,
    admission::Fields,
    context::Context,
    geometry::{SurveyedGeometry, TensorGeometry},
    surveyed::work::WorkLedger,
};
use numr::{dtype::DType, runtime::Runtime};

/// Simpson panels of a margin estimate. The margin steers a step length, so it needs no
/// certified enclosure.
const PANELS: usize = 256;

/// A sonic outflow margin `A u_b (d_s) - q` and the width of its enclosure, in m³/s.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::channel::flow) struct Margin {
    pub value: f64,
    pub width: f64,
}

impl Margin {
    /// The margin is certainly positive.
    pub(in crate::channel::flow) fn holds(&self) -> bool {
        self.value - 0.5 * self.width > 0.0
    }

    /// The margin is certainly negative: the target exceeds the sonic outflow.
    pub(in crate::channel::flow) fn lost(&self) -> bool {
        self.value + 0.5 * self.width < 0.0
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TraceClient<R>,
{
    /// The least sonic outflow margin of a surveyed reach, or `None`.
    ///
    /// Rectangular and triangular reaches return `None`, and so does an estimate that
    /// rejects. A `None` margin keeps the coordinator's plain interval halving.
    pub(in crate::channel::flow) fn capacity_margin(
        &self,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Option<Margin> {
        match &self.geometry {
            TensorGeometry::Surveyed(geometry) => self
                .surveyed_capacity_margin(geometry, fields, left, right)
                .ok()
                .flatten(),
            _ => None,
        }
    }

    /// The least sonic outflow margin over the `Discharge` ends of `fields`, or `None`.
    ///
    /// An end contributes only when its target exceeds the discharge at the interior depth,
    /// so the trace lies between the sonic depth and the interior depth. Ends whose interior
    /// or sonic root does not resolve contribute nothing.
    pub(in crate::channel::flow) fn surveyed_capacity_margin(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Option<Margin>, ChannelError> {
        let mut least: Option<Margin> = None;
        for (endpoint, policy) in [left, right].into_iter().enumerate() {
            if !matches!(policy, ChannelTensorBoundary::Discharge { .. }) {
                continue;
            }
            let Ok(interior) = self.surveyed_interior(geometry, fields, endpoint) else {
                continue;
            };
            let TraceRequest::Discharge { outward, .. } =
                self.surveyed_trace_request(&interior, policy)?
            else {
                continue;
            };
            let config = &geometry.riemann;
            let work = WorkLedger::new(config);
            let panels = config.max_simpson_panels.min(PANELS);
            work.begin_refinement(panels, 0).map_err(trace_error)?;
            let evaluation = Evaluation {
                config,
                panels,
                work: &work,
            };
            let (interior_t, sonic_slope) = self.surveyed_sonic_setup(&interior)?;
            if interior_t > 0.0 {
                let point =
                    self.surveyed_trace_at(&interior, &self.tensor(&[interior_t])?, &evaluation)?;
                let residual = self.surveyed_discharge_residual(&interior, &point, &outward)?;
                if residual.upper().item::<f64>()? <= 0.0 {
                    continue;
                }
            }
            let Ok(Ok(root)) =
                self.surveyed_sonic_root(&interior, &evaluation, interior_t, sonic_slope)
            else {
                continue;
            };
            let point =
                self.surveyed_trace_at(&interior, &self.tensor(&[root.upper])?, &evaluation)?;
            // The residual is q - A u_b, so the margin is its negated center.
            let residual = self.surveyed_discharge_residual(&interior, &point, &outward)?;
            let (lower, upper) = (
                residual.lower().item::<f64>()?,
                residual.upper().item::<f64>()?,
            );
            let margin = Margin {
                value: -0.5 * (lower + upper),
                width: upper - lower,
            };
            if margin.value.is_finite() && margin.width.is_finite() {
                least = Some(match least {
                    Some(kept) if kept.value <= margin.value => kept,
                    _ => margin,
                });
            }
        }
        Ok(least)
    }
}

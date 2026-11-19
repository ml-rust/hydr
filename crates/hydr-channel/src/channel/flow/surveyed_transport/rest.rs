//! Momentum rates of surveyed scalar end faces whose trace is the end cell at rest.
//!
//! An interior face between two equal cells carries the two-part point flux of that cell.
//! A scalar trace at rest is exactly the end cell's state, so its face takes the same point
//! flux. The end cell then sees the same momentum point on both faces, and its transfer
//! cancels exactly. Water keeps the trace's single-part rate.

use super::error::riemann_error;
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary,
    admission::Fields,
    boundary::DirectBoundaryFrame,
    conserved_point::ConservedPoint,
    conserved_update::{Parts, parts},
    context::Context,
    geometry::SurveyedGeometry,
    surveyed::work::PrismaticRiemannWork,
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms,
        roots::{
            BatchedBisectionAlgorithms, CertifiedBisectionAlgorithms,
            CertifiedUpperBracketAlgorithms,
        },
    },
};

/// The end cell's own point flux at each rest end face.
pub(super) struct RestMomentum<R: Runtime<DType = DType>> {
    /// Rest end faces, `[left, right]`.
    pub ends: [bool; 2],
    /// One momentum rate per rest end, left before right.
    pub momentum: Parts<R>,
    /// Work of the point Riemann solve of these faces.
    pub work: PrismaticRiemannWork,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + MidpointOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// The end cell's two-part point flux at each scalar end at rest, or `None` without one.
    ///
    /// Each rest face solves the point Riemann problem with the end cell on both sides. An
    /// interior face between two equal cells solves the same problem.
    pub(super) fn surveyed_rest_momentum(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
        frame: Option<&DirectBoundaryFrame<R>>,
    ) -> Result<Option<RestMomentum<R>>, ChannelError> {
        let ends = self.surveyed_rest_ends(fields, left, right, frame)?;
        if !ends.contains(&true) {
            return Ok(None);
        }
        let c = &self.client;
        let count = self.count();
        let mut cells = Vec::with_capacity(2);
        let mut faces = Vec::with_capacity(2);
        for (end, (cell, face)) in [(0, 0), (count - 1, count)].into_iter().enumerate() {
            if ends[end] {
                cells.push(cell);
                faces.push(face);
            }
        }
        let pick = |value: &Tensor<R>, rows: &[usize]| {
            let pieces = rows
                .iter()
                .map(|&row| value.narrow(0, row, 1))
                .collect::<Result<Vec<_>, _>>()?;
            let pieces: Vec<&Tensor<R>> = pieces.iter().collect();
            Ok::<_, ChannelError>(c.cat(&pieces, 0)?)
        };
        let point = |value: &ConservedPoint<R>| {
            Ok::<_, ChannelError>(ConservedPoint {
                high: pick(&value.high, &cells)?,
                low: pick(&value.low, &cells)?,
            })
        };
        let area = point(&fields.area_point)?;
        let discharge = point(&fields.discharge_point)?;
        let states = self.surveyed_face_points(&area, &discharge, &area, &discharge)?;
        let result = geometry
            .section
            .riemann_flux_points_tensor(
                &pick(&geometry.face_owners, &faces)?,
                &states.left_area,
                &states.left_discharge,
                &states.right_area,
                &states.right_discharge,
                &geometry.riemann,
            )
            .map_err(riemann_error)?;
        Ok(Some(RestMomentum {
            ends,
            momentum: parts(result.momentum_flux_m4_per_s2()),
            work: result.work(),
        }))
    }

    /// Copy of the face momentum rates with each rest end face set to its point flux.
    pub(super) fn surveyed_rest_parts(
        &self,
        momentum: Parts<R>,
        rest: &RestMomentum<R>,
    ) -> Result<Parts<R>, ChannelError> {
        let mut momentum = momentum;
        let mut lane = 0;
        for (end, face) in [(0, 0), (1, self.count())] {
            if !rest.ends[end] {
                continue;
            }
            let take = |value: &Tensor<R>| value.narrow(0, lane, 1);
            momentum = Parts {
                high: self.replace_face(&momentum.high, face, &take(&rest.momentum.high)?)?,
                low: self.replace_face(&momentum.low, face, &take(&rest.momentum.low)?)?,
                bound: self.replace_face(&momentum.bound, face, &take(&rest.momentum.bound)?)?,
            };
            lane += 1;
        }
        Ok(momentum)
    }

    /// Scalar end faces whose trace is exactly the end cell's own state, `[left, right]`.
    ///
    /// The end cell must hold exactly zero discharge in both parts. Every trace lies on the
    /// end cell's wave curve `u_b = u_i - f_i(A_b)`, where `f_i` is strictly monotone and
    /// `f_i(A_i) = 0`. So `u_b = u_i = 0` holds only at `A_b = A_i`. A `Discharge` end with a
    /// zero target solves `A_b u_b = 0` with `A_b > 0`, so `u_b = 0`. Every other scalar end
    /// needs a certified trace velocity of exactly `[0, 0]`: a zero center with a zero radius.
    fn surveyed_rest_ends(
        &self,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
        frame: Option<&DirectBoundaryFrame<R>>,
    ) -> Result<[bool; 2], ChannelError> {
        let mut ends = [false; 2];
        let Some(traces) = frame.and_then(|frame| frame.traces.as_deref()) else {
            return Ok(ends);
        };
        let velocity = traces.outward_velocity_m_per_s.try_to_vec::<f64>()?;
        let radius = traces
            .velocity_compatibility_abs_bound_m_per_s
            .try_to_vec::<f64>()?;
        let high = fields.discharge_point.high.try_to_vec::<f64>()?;
        let low = fields.discharge_point.low.try_to_vec::<f64>()?;
        for (lane, &endpoint) in traces.endpoints.iter().enumerate() {
            let cell = if endpoint == 0 { 0 } else { self.count() - 1 };
            if high[cell] != 0.0 || low[cell] != 0.0 {
                continue;
            }
            let policy = if endpoint == 0 { left } else { right };
            ends[endpoint] = match policy {
                ChannelTensorBoundary::Discharge { discharge_m3_per_s } => {
                    discharge_m3_per_s.item::<f64>()? == 0.0
                }
                _ => velocity[lane] == 0.0 && radius[lane] == 0.0,
            };
        }
        Ok(ends)
    }
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::{
        ChannelBoundary, ChannelFluxPort, ChannelForcingInterval, ChannelReach,
        surveyed_boundary::fixture::reach_cells,
        surveyed_update::tests::{Client, TestResult, cpu, moving_reach},
    };
    use numr::dtype::DType;
    use numr::runtime::Runtime;
    use numr::runtime::cpu::CpuRuntime;

    /// Intervals of each rest run.
    const INTERVALS: usize = 50;
    /// Length of each interval, in seconds.
    const SPAN: f64 = 0.125;

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// Every admitted end policy at rest against an end cell of depth 1 m on a bed at 0 m.
    ///
    /// `port` holds the reference pressure flux `g I(A)` of a zero-water flux port.
    fn policies(port: Option<(f64, f64)>) -> TestResult<Vec<ChannelBoundary>> {
        let reservoir = |entrance_loss_coefficient| ChannelBoundary::QuiescentReservoir {
            elevation_m: 1.0,
            entrance_loss_coefficient,
        };
        let mut policies = vec![
            ChannelBoundary::Wall,
            ChannelBoundary::Exterior {
                depth_m: 1.0,
                velocity_m_per_s: 0.0,
            },
            ChannelBoundary::Stage { elevation_m: 1.0 },
            ChannelBoundary::Discharge {
                discharge_m3_per_s: 0.0,
            },
            reservoir(0.0),
            reservoir(1.0),
        ];
        if let Some((pressure, invariant)) = port {
            policies.push(ChannelBoundary::FluxPort(ChannelFluxPort::new(
                0.0,
                pressure,
                2.0 * invariant,
                Some(invariant),
                "Zero-water port at the reference rest pressure",
            )?));
        }
        Ok(policies)
    }

    /// Runs every ordered pair of `policies` and checks each cell bit for bit after each interval.
    fn hold_rest<R>(origin: &ChannelReach<R>, policies: &[ChannelBoundary]) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let start = origin.snapshot_data()?;
        let cells = origin.state()?;
        for left in policies {
            for right in policies {
                let mut current = origin.clone();
                for index in 0..INTERVALS {
                    let end = (index + 1) as f64 * SPAN;
                    let forcing = ChannelForcingInterval {
                        start_s: current.time(),
                        end_s: end,
                        left: left.clone(),
                        right: right.clone(),
                        lateral_inputs: Vec::new(),
                    };
                    current = current.step(end, &forcing)?.next;
                    let pair = format!("{left:?} / {right:?}, interval {index}");
                    let data = current.snapshot_data()?;
                    for (actual, expected) in [
                        (&data.volume_m3, &start.volume_m3),
                        (&data.volume_low_m3, &start.volume_low_m3),
                        (
                            &data.longitudinal_momentum_m4_per_s,
                            &start.longitudinal_momentum_m4_per_s,
                        ),
                        (
                            &data.longitudinal_momentum_low_m4_per_s,
                            &start.longitudinal_momentum_low_m4_per_s,
                        ),
                    ] {
                        assert_eq!(bits(actual), bits(expected), "{pair}");
                    }
                    for (cell, before) in current.state()?.iter().zip(&cells) {
                        assert_eq!(cell.volume_m3.to_bits(), before.volume_m3.to_bits());
                        assert_eq!(
                            cell.discharge_m3_per_s.to_bits(),
                            before.discharge_m3_per_s.to_bits(),
                            "{pair}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Eight rectangle cells of width 2 m at rest at depth 1 m hold exactly still.
    ///
    /// Frozen criteria, for every ordered pair of `Wall`, `Exterior` at rest, `Stage(1 m)`,
    /// `Discharge(0)`, `QuiescentReservoir(1 m, k = 0)`, `QuiescentReservoir(1 m, k = 1)`, and a
    /// zero-water `FluxPort` at the reference pressure `g I(A) = g`:
    /// - over 50 intervals of 0.125 s, every cell's area and discharge stay bit-identical to
    ///   the start, both parts of each
    fn rectangle_rest<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let origin = moving_reach::<R>(client, &[10.0; 8], &[0.0; 8], &[0.0; 8])?;
        let gravity = origin.definition().config.gravity;
        // The width-2 rectangle at depth 1 m holds `I(A) = 1`, so `g I(A) = g` exactly.
        let invariant = 2.0 * gravity.sqrt();
        hold_rest(&origin, &policies(Some((gravity, invariant)))?)
    }

    /// Eight three-band cells at rest at depth 1 m hold exactly still.
    ///
    /// Frozen criteria, for every ordered pair of `Wall`, `Exterior` at rest, `Stage(1 m)`,
    /// `Discharge(0)`, `QuiescentReservoir(1 m, k = 0)`, and `QuiescentReservoir(1 m, k = 1)`:
    /// - over 50 intervals of 0.125 s, every cell's area and discharge stay bit-identical to
    ///   the start, both parts of each
    ///
    /// The three-band rest pressure `g I(A)` has no binary64 value. Interior faces carry its
    /// two-part point, and a flux port carries one F64 rate. No port rate equals that point, so
    /// the matrix leaves the port out.
    fn three_band_rest<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        // The three-band section holds 3 m² at depth 1 m.
        let origin = reach_cells::<R>(client, 8, 3.0, 0.0, 0.0)?;
        hold_rest(&origin, &policies(None)?)
    }

    #[test]
    fn cpu_surveyed_rectangle_rest_holds_bit_for_bit_with_every_end_pair() -> TestResult {
        rectangle_rest::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_surveyed_three_band_rest_holds_bit_for_bit_with_every_end_pair() -> TestResult {
        three_band_rest::<CpuRuntime>(cpu())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

        #[test]
        fn cuda_surveyed_rectangle_rest_holds_bit_for_bit_with_every_end_pair() -> TestResult {
            rectangle_rest::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_surveyed_three_band_rest_holds_bit_for_bit_with_every_end_pair() -> TestResult {
            three_band_rest::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }
    }
}

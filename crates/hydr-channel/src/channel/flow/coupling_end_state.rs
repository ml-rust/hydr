//! Private coordinator read: the accepted-start state of one rectangular reach end cell.

use super::{ChannelEnd, ChannelError, ChannelGeometry, ChannelReach};
use crate::channel::CrossSectionDefinition;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps,
    TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Accepted-start state of a rectangular end cell, as a flux port ghost needs it.
///
/// Every tensor is F64 with shape `[1]` on the reach device. The values equal the
/// `state()` fields of that cell bit for bit.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct ChannelCouplingEndState<R: Runtime<DType = DType>> {
    /// End cell index in cell order.
    pub cell: usize,
    /// End cell bed elevation in the reach datum, in m.
    pub bed_elevation_m: f64,
    /// End cell width, in m.
    pub width_m: f64,
    /// Wetted area `volume / length`, in m².
    pub area_m2: Tensor<R>,
    /// Discharge along the increasing reach coordinate, in m³/s.
    pub discharge_m3_per_s: Tensor<R>,
    /// Depth `area / width`, in m.
    pub depth_m: Tensor<R>,
    /// Discharge per width `discharge / width` along the increasing reach coordinate, in m²/s.
    pub discharge_per_width_m2_per_s: Tensor<R>,
}

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    /// Accepted-start area, discharge, depth and discharge per width of the `end` cell.
    ///
    /// Only a rectangular section has one width that a strip of surface faces can match.
    /// A surveyed reach qualifies when its profile is a rectangle. Its depth is `area / width`.
    ///
    /// # Errors
    /// Rejects triangular reaches and surveyed profiles that are not rectangles with
    /// `port-section-mismatch`. Returns field recovery errors.
    #[doc(hidden)]
    pub fn coupling_end_state(
        &self,
        end: ChannelEnd,
    ) -> Result<ChannelCouplingEndState<R>, ChannelError> {
        let cell = match end {
            ChannelEnd::Left => 0,
            ChannelEnd::Right => self.context.count() - 1,
        };
        let (bed_elevation_m, width_m, surveyed) = match &self.definition().geometry {
            ChannelGeometry::Rectangular { cells } => {
                (cells[cell].bed_elevation_m, cells[cell].width_m, false)
            }
            ChannelGeometry::PrismaticSurveyed { profile, .. } => {
                let width = surveyed_rectangle_width(profile).ok_or_else(|| {
                    ChannelError::input(
                        "port-section-mismatch: a reach-end port needs a rectangular surveyed section; use a rectangular profile or connect this reach through bank links",
                    )
                })?;
                (
                    self.definition().geometry.cell(cell).bed_elevation_m,
                    width,
                    true,
                )
            }
            ChannelGeometry::PrismaticTriangular { .. } => {
                return Err(ChannelError::input(
                    "port-section-mismatch: a reach-end port needs a rectangular end cell of the port width; triangular geometry has none, so use a rectangular reach or connect this reach through bank links",
                ));
            }
        };
        let c = &self.context.client;
        let fields = self
            .context
            .fields(&self.accepted.volume, &self.accepted.momentum)?;
        let area = fields.area.narrow(0, cell, 1)?.contiguous()?;
        let depth = if surveyed {
            c.div_scalar(&area, width_m)?
        } else {
            fields.depth.narrow(0, cell, 1)?.contiguous()?
        };
        let discharge = fields.discharge.narrow(0, cell, 1)?.contiguous()?;
        let discharge_per_width = c.div_scalar(&discharge, width_m)?;
        Ok(ChannelCouplingEndState {
            cell,
            bed_elevation_m,
            width_m,
            area_m2: area,
            discharge_m3_per_s: discharge,
            depth_m: depth,
            discharge_per_width_m2_per_s: discharge_per_width,
        })
    }
}

/// Bottom width `b` of a surveyed profile that is a rectangle, else `None`.
///
/// The minimum-elevation points form one contiguous run of at least two points.
/// Points before the run share its first station and points after it share its last.
/// Both end points reach the maximum stage.
///
/// Coordinators call this to check a port's width before a reach state exists.
/// `coupling_end_state` uses the same test.
#[doc(hidden)]
pub fn surveyed_rectangle_width(profile: &CrossSectionDefinition) -> Option<f64> {
    let points = &profile.points;
    let minimum = points
        .iter()
        .map(|point| point.elevation_m)
        .reduce(f64::min)?;
    let first = points.iter().position(|p| p.elevation_m == minimum)?;
    let last = points.iter().rposition(|p| p.elevation_m == minimum)?;
    if last == first
        || points[first..=last]
            .iter()
            .any(|p| p.elevation_m != minimum)
    {
        return None;
    }
    let left = points[first].station_m;
    let right = points[last].station_m;
    let walls = points[..first].iter().all(|p| p.station_m == left)
        && points[last + 1..].iter().all(|p| p.station_m == right);
    let top = profile.maximum_stage_m;
    let reaches = points.first()?.elevation_m == top && points.last()?.elevation_m == top;
    let width = right - left;
    (walls && reaches && width > 0.0).then_some(width)
}

#[cfg(test)]
mod tests {
    use super::super::{
        ChannelConfig, ChannelInitialState, ChannelReachDefinition, RectangularChannelCell,
        TriangularChannelCell,
    };
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn client() -> CpuClient {
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
    }

    fn rectangular() -> ChannelReachDefinition {
        let cell = |id: &str, width: f64, bed: f64| RectangularChannelCell {
            id: id.into(),
            length_m: 7.0,
            width_m: width,
            bed_elevation_m: bed,
            maximum_depth_m: 5.0,
            manning_n: 0.0,
        };
        ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "coupling end state fixture".into(),
            geometry: ChannelGeometry::Rectangular {
                cells: vec![
                    cell("a", 3.0, 1.25),
                    cell("b", 2.5, 1.0),
                    cell("c", 1.75, 0.5),
                ],
            },
            lateral_bindings: Vec::new(),
            config: ChannelConfig::default(),
        }
    }

    #[test]
    fn end_states_equal_state_fields() -> TestResult {
        let reach = ChannelReach::<CpuRuntime>::new(
            client(),
            rectangular(),
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![13.0, 9.0, 11.3],
                longitudinal_momentum_m4_per_s: vec![-2.1, 0.4, 6.7],
            },
        )?;
        let state = reach.state()?;
        let ChannelGeometry::Rectangular { cells } = &reach.definition().geometry else {
            return Err("the fixture is rectangular".into());
        };
        for (end, cell) in [(ChannelEnd::Left, 0), (ChannelEnd::Right, 2)] {
            let read = reach.coupling_end_state(end)?;
            let scalar = |tensor: &Tensor<CpuRuntime>| -> Result<f64, Box<dyn std::error::Error>> {
                assert_eq!(tensor.shape(), [1]);
                Ok(tensor.try_to_vec::<f64>()?[0])
            };
            let declared = &cells[cell];
            let result = &state[cell];
            assert_eq!(read.cell, cell);
            assert_eq!(
                read.bed_elevation_m.to_bits(),
                declared.bed_elevation_m.to_bits()
            );
            assert_eq!(read.width_m.to_bits(), declared.width_m.to_bits());
            assert_eq!(
                scalar(&read.area_m2)?.to_bits(),
                (result.volume_m3 / declared.length_m).to_bits()
            );
            assert_eq!(scalar(&read.depth_m)?.to_bits(), result.depth_m.to_bits());
            assert_eq!(
                scalar(&read.discharge_m3_per_s)?.to_bits(),
                result.discharge_m3_per_s.to_bits()
            );
            assert_eq!(
                scalar(&read.discharge_per_width_m2_per_s)?.to_bits(),
                (result.discharge_m3_per_s / declared.width_m).to_bits()
            );
        }
        Ok(())
    }

    #[test]
    fn dry_end_reads_zero() -> TestResult {
        let reach = ChannelReach::<CpuRuntime>::new(
            client(),
            rectangular(),
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![0.0, 9.0, 11.3],
                longitudinal_momentum_m4_per_s: vec![0.0, 0.4, 6.7],
            },
        )?;
        let read = reach.coupling_end_state(ChannelEnd::Left)?;
        for tensor in [
            &read.area_m2,
            &read.depth_m,
            &read.discharge_m3_per_s,
            &read.discharge_per_width_m2_per_s,
        ] {
            assert_eq!(tensor.try_to_vec::<f64>()?, vec![0.0]);
        }
        Ok(())
    }

    #[test]
    fn triangular_end_state_rejects() -> TestResult {
        let cell = |id: &str| TriangularChannelCell {
            id: id.into(),
            length_m: 5.0,
            maximum_depth_m: 4.0,
            manning_n: 0.0,
        };
        let reach = ChannelReach::<CpuRuntime>::new(
            client(),
            ChannelReachDefinition {
                vertical_datum: "synthetic datum".into(),
                provenance: "triangular end state fixture".into(),
                geometry: ChannelGeometry::PrismaticTriangular {
                    left_bank_slope_h_per_v: 1.0,
                    right_bank_slope_h_per_v: 2.0,
                    bed_elevation_m: 0.0,
                    cells: vec![cell("a"), cell("b")],
                },
                lateral_bindings: Vec::new(),
                config: ChannelConfig::default(),
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![5.0, 5.0],
                longitudinal_momentum_m4_per_s: vec![0.0, 0.0],
            },
        )?;
        let error = reach
            .coupling_end_state(ChannelEnd::Right)
            .err()
            .ok_or("a triangular end must reject")?;
        assert!(
            matches!(error, ChannelError::InvalidInput { .. }),
            "{error}"
        );
        let message = error.to_string();
        assert!(message.contains("port-section-mismatch"), "{message}");
        assert!(message.contains("rectangular end cell"), "{message}");
        Ok(())
    }
}

//! Flux port end faces: direct rates, the unsigned speed bound and the exterior invariant.
//!
//! A port supplies its rates along the outward normal. The face arrays hold rates along the
//! increasing reach coordinate. The water rate takes the outward sign `σ`, `-1` left and `+1`
//! right. The normal momentum flux keeps its sign at both ends, so the end cell gains
//! `-σ · outward_momentum_rate`.

use super::{EndpointInterior, PortFront};
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary, admission::Fields, context::Context,
    geometry::TensorGeometry, triangular::envelope::lower,
};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};

/// Face rates of one flux port endpoint, each of shape `[1]`.
pub(super) struct PortLane<R: Runtime<DType = DType>> {
    pub endpoint: usize,
    /// Water rate along the increasing reach coordinate, in m³/s.
    pub longitudinal: Tensor<R>,
    /// Normal momentum flux, including pressure, in m⁴/s².
    pub momentum: Tensor<R>,
    /// Unsigned bound on the face signal speeds, in m/s.
    pub speed: Tensor<R>,
    /// Supplied rates carry no rounding bound and no flux bracket width.
    pub zero: Tensor<R>,
    invariant: Option<Tensor<R>>,
}

impl<R: Runtime<DType = DType>> PortLane<R> {
    pub fn front(self) -> PortFront<R> {
        PortFront {
            endpoint: self.endpoint,
            invariant: self.invariant,
        }
    }
}

/// Outward normal sign of `endpoint`: `-1` left, `+1` right.
fn outward(endpoint: usize) -> f64 {
    if endpoint == 0 { -1.0 } else { 1.0 }
}

fn side(endpoint: usize) -> &'static str {
    if endpoint == 0 { "left" } else { "right" }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    /// Face rates of each flux port endpoint, in endpoint order.
    pub(super) fn port_lanes(
        &self,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Vec<PortLane<R>>, ChannelError> {
        let c = &self.client;
        let mut lanes = Vec::with_capacity(2);
        for (endpoint, boundary) in [left, right].into_iter().enumerate() {
            let ChannelTensorBoundary::FluxPort(port) = boundary else {
                continue;
            };
            let water = port.outward_water_rate_m3_per_s();
            // Negation is exact.
            let longitudinal = c.mul_scalar(water, outward(endpoint))?;
            lanes.push(PortLane {
                endpoint,
                longitudinal,
                momentum: port.outward_momentum_rate_m4_per_s2().clone(),
                speed: port.signal_speed_m_per_s().clone(),
                zero: self.constant(water, 0.0)?,
                invariant: port.exterior_invariant_m_per_s().cloned(),
            });
        }
        Ok(lanes)
    }

    /// Rejects a port speed below the end cell's `|u| + c` and export from a dry end cell.
    ///
    /// `fields` holds the accepted-start cell averages. `interior` holds the end-cell side of
    /// each end face. The speed must bound both states. The required speed is rounded down, so
    /// a provider that rounds the same characteristic speed is not rejected.
    pub(in crate::channel::flow) fn check_flux_ports(
        &self,
        fields: &Fields<R>,
        interior: &EndpointInterior<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<(), ChannelError> {
        let c = &self.client;
        let count = self.count();
        let gravity = self.definition.config.gravity;
        // Rectangular c = sqrt(g h). Triangular c = sqrt(g A / T) = sqrt(g h / 2).
        let sound_factor = match &self.geometry {
            TensorGeometry::Rectangular(_) => gravity,
            TensorGeometry::Triangular(_) => gravity * 0.5,
            TensorGeometry::Surveyed(_) => {
                return Err(ChannelError::input(
                    "surveyed FluxPort endpoint is unsupported: supply Wall or complete Exterior depth and velocity",
                ));
            }
        };
        for (endpoint, boundary) in [left, right].into_iter().enumerate() {
            let ChannelTensorBoundary::FluxPort(port) = boundary else {
                continue;
            };
            let cell = if endpoint == 0 { 0 } else { count - 1 };
            let characteristic = |depth: &Tensor<R>, velocity: &Tensor<R>| {
                let sound = c.sqrt(&c.mul_scalar(depth, sound_factor)?)?;
                lower::<R>(c, &c.add(&c.abs(velocity)?, &sound)?)
            };
            let average = characteristic(
                &fields.depth.narrow(0, cell, 1)?,
                &fields.velocity.narrow(0, cell, 1)?,
            )?;
            let face = characteristic(
                &interior.depth.narrow(0, endpoint, 1)?,
                &interior.velocity.narrow(0, endpoint, 1)?,
            )?;
            let required = c.maximum(&average, &face)?;
            self.finite(&required, "flux port end cell characteristic speed")?;
            let required = required.item::<f64>()?;
            let speed = port.signal_speed_m_per_s().item::<f64>()?;
            if speed < required {
                return Err(ChannelError::input(format!(
                    "flux port {} speed={speed} is below the end cell |u|+c={required}: supply a signal-speed bound of at least the end cell characteristic speed",
                    side(endpoint)
                ))
                .at(cell));
            }
            let water = port.outward_water_rate_m3_per_s().item::<f64>()?;
            let area = fields.area.narrow(0, cell, 1)?.item::<f64>()?;
            if water > 0.0 && area <= 0.0 {
                return Err(ChannelError::input(format!(
                    "flux port {} exports water={water} m³/s from a dry end cell: supply zero or entering water at a dry end",
                    side(endpoint)
                ))
                .at(cell));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::{ChannelReach, ChannelTensorBoundary, ChannelTensorFluxPort};
    use crate::channel::{
        ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReachDefinition,
        RectangularChannelCell,
    };
    use numr::runtime::RuntimeClient;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    use numr::tensor::Tensor;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn cpu() -> CpuClient {
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
    }

    fn port(
        client: &CpuClient,
        water: f64,
        momentum: f64,
        speed: f64,
        invariant: Option<f64>,
    ) -> Result<ChannelTensorBoundary<CpuRuntime>, Box<dyn std::error::Error>> {
        let value = |v: f64| Tensor::from_slice(&[v], &[1], client.device());
        Ok(ChannelTensorBoundary::FluxPort(Box::new(
            ChannelTensorFluxPort::new(
                value(water)?,
                value(momentum)?,
                value(speed)?,
                invariant.map(value).transpose()?,
                "frame fixture",
            )?,
        )))
    }

    /// Two resting unit-depth cells of unit width and length, with gravity 1.
    fn reach(client: CpuClient) -> Result<ChannelReach<CpuRuntime>, Box<dyn std::error::Error>> {
        let cell = |id: &str| RectangularChannelCell {
            id: id.into(),
            length_m: 1.0,
            width_m: 1.0,
            bed_elevation_m: 0.0,
            maximum_depth_m: 10.0,
            manning_n: 0.0,
        };
        Ok(ChannelReach::new(
            client,
            ChannelReachDefinition {
                vertical_datum: "synthetic datum".into(),
                provenance: "flux port frame fixture".into(),
                geometry: ChannelGeometry::Rectangular {
                    cells: vec![cell("a"), cell("b")],
                },
                lateral_bindings: vec![],
                config: ChannelConfig {
                    gravity: 1.0,
                    ..ChannelConfig::default()
                },
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![1.0, 1.0],
                longitudinal_momentum_m4_per_s: vec![0.0, 0.0],
            },
        )?)
    }

    #[test]
    fn port_faces_carry_signed_water_and_unsigned_speed() -> TestResult {
        let client = cpu();
        let reach = reach(client.clone())?;
        let context = &reach.context;
        let left = port(&client, -0.25, 0.75, 2.0, Some(3.0))?;
        let right = port(&client, 0.125, 0.5, 4.0, None)?;
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let flux = context.flux(&fields, &left, &right)?;
        let water = flux.water.try_to_vec::<f64>()?;
        let momentum = flux.momentum.try_to_vec::<f64>()?;
        // Water takes the outward sign. Normal momentum keeps its sign at both ends.
        assert_eq!([water[0], water[2]], [0.25, 0.125]);
        assert_eq!([momentum[0], momentum[2]], [0.75, 0.5]);
        let rightward = flux.speed.rightward.try_to_vec::<f64>()?;
        let leftward = flux.speed.leftward.try_to_vec::<f64>()?;
        assert_eq!([rightward[0], leftward[0]], [2.0, 2.0]);
        assert_eq!([rightward[2], leftward[2]], [4.0, 4.0]);
        // The end cell side pressure correction is zero at both port faces.
        assert_eq!(flux.right_pressure.try_to_vec::<f64>()?[0], 0.0);
        assert_eq!(flux.left_pressure.try_to_vec::<f64>()?[2], 0.0);
        assert!(flux.boundary_traces.is_none());
        let reservation = context.outgoing_rate(&flux.water)?.try_to_vec::<f64>()?;
        assert_eq!(reservation, [0.0, 0.125]);
        Ok(())
    }

    #[test]
    fn a_slow_port_rejects_against_the_end_cell() -> TestResult {
        let client = cpu();
        let reach = reach(client.clone())?;
        let context = &reach.context;
        let wall = ChannelTensorBoundary::Wall;
        // At unit depth with gravity 1 the end cell has |u| + c = 1.
        let slow = port(&client, 0.0, 0.5, 0.5, Some(2.0))?;
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let error = context
            .flux(&fields, &wall, &slow)
            .err()
            .ok_or("a slow port was admitted")?;
        assert!(
            error.to_string().contains("flux port right speed=0.5"),
            "{error}"
        );
        let exact = port(&client, 0.0, 0.5, 1.0, Some(2.0))?;
        context.flux(&fields, &wall, &exact)?;
        Ok(())
    }
}

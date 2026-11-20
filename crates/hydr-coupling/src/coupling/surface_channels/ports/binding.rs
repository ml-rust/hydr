//! Per-attempt port binding: host flux receipts, reach flux ports, and ported reservations.
//!
//! The surface stage holds one HLL flux per port face. Each ported reach end then takes a
//! `FluxPort` built from its port's summed rates. The signal speed it receives is the largest of
//! the port's HLL face bounds and `|u| + c` of its reach end and its surface cells, raised by
//! `4 ε` of its magnitude. HLL bounds can fall below `|u| + c` across a strong shock, and the
//! reach rejects a speed below its own `|u| + c`. A larger speed only shortens the CFL limit.
//!
//! The exterior invariant is the largest into-reach `u + 2c` over the wet surface face cells,
//! raised the same way. It is `None` when every surface face cell is dry.

use super::super::SurfaceChannelError;
use super::super::context::{Context, Ops};
use super::super::start::Start;
use super::ghosts::PortEnd;
use super::receipt::{FaceSide, PortFlux, face_regime, port_regime};
use super::{CheckedPort, end_slot};
use crate::channel::ChannelEnd;
use crate::channel::{ChannelReach, ChannelTensorBoundary, ChannelTensorFluxPort};
use hydr_channel::internal::ChannelCouplingReservation;
use hydr_solver::internal::{PortFluxes, SurfaceCouplingStage};
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Relative margin that raises a port speed or invariant, in units of `f64::EPSILON`.
const SPEED_MARGIN_ULPS: f64 = 4.0;

/// Provider, compatible state, and speed basis of every coordinator flux port.
const PORT_PROVENANCE: &str = "surface/channel coordinator HLL port: accepted-start reach end cell against post-source surface port faces; speed is the largest HLL face bound and |u|+c of both states, raised by 4 epsilon";

/// Port receipts and ported reach reservations of one attempted duration.
pub(in super::super) struct PortBinding<R: Runtime<DType = DType>> {
    /// Per setup reach: the reservation of a ported reach for this attempt.
    reservations: Vec<Option<ChannelCouplingReservation<R>>>,
    /// Smallest CFL or face depletion limit over every ported reach, in seconds.
    pub limit_s: f64,
    /// Host flux receipt of every port, in port order.
    pub fluxes: Vec<PortFlux>,
    /// Summed port water rate into the surface and out of it, as the surface counts it, in m³/s.
    pub surface_rate_m3_per_s: [f64; 2],
    /// Flux port endpoint of every port, in port order.
    boundaries: Vec<ChannelTensorBoundary<R>>,
}

impl<R: Runtime<DType = DType>> PortBinding<R> {
    /// The flux port endpoint of `port`, in port order.
    pub fn boundary(&self, port: usize) -> Option<&ChannelTensorBoundary<R>> {
        self.boundaries.get(port)
    }

    /// The attempt reservation of `reach`: its port reservation, else its start reservation.
    pub fn reservation<'a>(
        &'a self,
        start: &'a Start<R>,
        reach: usize,
    ) -> Result<&'a ChannelCouplingReservation<R>, SurfaceChannelError> {
        self.reservations
            .get(reach)
            .and_then(Option::as_ref)
            .or_else(|| start.reservations.get(reach).and_then(Option::as_ref))
            .ok_or_else(|| SurfaceChannelError::Numerical("a reach lacks its reservation".into()))
    }
}

/// `value` raised by [`SPEED_MARGIN_ULPS`] `ε` of its magnitude and one subnormal quantum.
fn raised(value: f64) -> f64 {
    value + (SPEED_MARGIN_ULPS * f64::EPSILON * value.abs() + f64::from_bits(1))
}

/// `|u| + sqrt(g h)` of a wet state, zero for a dry one.
fn characteristic(depth: f64, velocity: f64, gravity: f64) -> f64 {
    if depth > 0.0 {
        velocity.abs() + (gravity * depth).sqrt()
    } else {
        0.0
    }
}

/// Host values of every port face lane and every port, read from the device once.
struct HostFluxes {
    faces: usize,
    ports: usize,
    values: Vec<f64>,
}

/// Face lane fields in read order.
#[derive(Clone, Copy)]
enum Face {
    Water,
    Width,
    Speed,
    Lower,
    Upper,
    Depth,
    Discharge,
}

/// Port fields in read order, after every face field.
#[derive(Clone, Copy)]
enum Port {
    Water,
    Momentum,
    SurfaceX,
    SurfaceY,
    ReactionX,
    ReactionY,
}

const FACE_FIELDS: usize = 7;

impl HostFluxes {
    fn read<R: Runtime<DType = DType>>(
        c: &R::Client,
        fluxes: &PortFluxes<R>,
    ) -> Result<Self, SurfaceChannelError>
    where
        R::Client: ShapeOps<R>,
    {
        let [surface_x, surface_y] = &fluxes.surface_momentum_rate_m4_per_s2;
        let [reaction_x, reaction_y] = &fluxes.junction_reaction_rate_m4_per_s2;
        let parts = [
            &fluxes.face_water_into_surface_m2_per_s,
            &fluxes.face_width_m,
            &fluxes.face_signal_speed_m_per_s,
            &fluxes.face_lower_wave_into_surface_m_per_s,
            &fluxes.face_upper_wave_into_surface_m_per_s,
            &fluxes.face_surface_depth_m,
            &fluxes.face_surface_discharge_into_surface_m2_per_s,
            &fluxes.reach_outward_water_rate_m3_per_s,
            &fluxes.reach_outward_momentum_rate_m4_per_s2,
            surface_x,
            surface_y,
            reaction_x,
            reaction_y,
        ];
        Ok(Self {
            faces: fluxes.face_width_m.numel(),
            ports: fluxes.reach_outward_water_rate_m3_per_s.numel(),
            values: c.cat(&parts, 0)?.try_to_vec::<f64>()?,
        })
    }

    fn get(&self, index: usize) -> Result<f64, SurfaceChannelError> {
        self.values
            .get(index)
            .copied()
            .ok_or_else(|| SurfaceChannelError::Numerical("port flux read lacks a value".into()))
    }

    fn face(&self, field: Face, lane: usize) -> Result<f64, SurfaceChannelError> {
        self.get(field as usize * self.faces + lane)
    }

    fn port(&self, field: Port, port: usize) -> Result<f64, SurfaceChannelError> {
        self.get(FACE_FIELDS * self.faces + field as usize * self.ports + port)
    }
}

/// Host flux receipt of `port`, whose face lanes start at `first`.
fn port_flux(
    host: &HostFluxes,
    index: usize,
    port: &CheckedPort,
    end: PortEnd,
    first: usize,
) -> Result<PortFlux, SurfaceChannelError> {
    let g = port.gravity;
    let ghost_bed = port.ghost_bed_m;
    let mut speed = characteristic(end.depth_m, end.outward_velocity_m_per_s, g);
    let mut invariant: Option<f64> = None;
    let mut rates = Vec::with_capacity(port.faces.len());
    let mut regimes = Vec::with_capacity(port.faces.len());
    for (offset, &surface_bed) in port.face_beds_m.iter().enumerate() {
        let lane = first + offset;
        let depth = host.face(Face::Depth, lane)?;
        let velocity = if depth > 0.0 {
            host.face(Face::Discharge, lane)? / depth
        } else {
            0.0
        };
        speed = speed
            .max(host.face(Face::Speed, lane)?)
            .max(characteristic(depth, velocity, g));
        if depth > 0.0 {
            let into_reach = -velocity + 2.0 * (g * depth).sqrt();
            invariant = Some(invariant.map_or(into_reach, |held| held.max(into_reach)));
        }
        rates.push(host.face(Face::Water, lane)? * host.face(Face::Width, lane)?);
        let top = ghost_bed.max(surface_bed);
        let ghost = FaceSide {
            depth_m: (end.depth_m - (top - ghost_bed)).max(0.0),
            velocity_m_per_s: end.outward_velocity_m_per_s,
        };
        let surface = FaceSide {
            depth_m: (depth - (top - surface_bed)).max(0.0),
            velocity_m_per_s: velocity,
        };
        let bounds = [host.face(Face::Lower, lane)?, host.face(Face::Upper, lane)?];
        regimes.push(face_regime(ghost, surface, bounds, g));
    }
    Ok(PortFlux {
        face_rate_m3_per_s: rates,
        outward_water_rate_m3_per_s: host.port(Port::Water, index)?,
        outward_momentum_rate_m4_per_s2: host.port(Port::Momentum, index)?,
        surface_momentum_rate_m4_per_s2: [
            host.port(Port::SurfaceX, index)?,
            host.port(Port::SurfaceY, index)?,
        ],
        junction_reaction_rate_m4_per_s2: [
            host.port(Port::ReactionX, index)?,
            host.port(Port::ReactionY, index)?,
        ],
        signal_speed_m_per_s: if speed > 0.0 { raised(speed) } else { speed },
        exterior_invariant_m_per_s: invariant.map(raised),
        regime: port_regime(&regimes),
    })
}

/// Whether a still port hands its reach end the reach's own `Wall` for this attempt.
///
/// A port moves no water in the attempt when its reach rate and every face rate are exactly
/// zero. If the end cell of a surveyed reach also rests in both discharge parts, the end takes
/// its `Wall` face. At rest a wall face mirrors the end cell onto itself, so it carries the
/// two-part rest pressure of the interior faces, and the end cell holds still bit for bit.
/// One F64 port rate cannot carry that two-part value.
///
/// The reach receipt records the wall flux the reach applies, both parts. The port receipt
/// keeps the coordinator's port rates, so a surveyed rectangle and a rectangular reach of the
/// same start state give the same port receipt.
///
/// Rectangular reaches keep their flux port. Their interior faces carry one F64 pressure,
/// which the port rate already matches at rest.
fn rest_wall<R: Runtime<DType = DType>>(
    reach: &ChannelReach<R>,
    port: &CheckedPort,
    flux: &PortFlux,
) -> Result<bool, crate::channel::ChannelError>
where
    R::Client: Ops<R>,
{
    let still = flux.outward_water_rate_m3_per_s == 0.0
        && flux.face_rate_m3_per_s.iter().all(|&rate| rate == 0.0);
    Ok(still && reach.coupling_rest_wall(port.end)?.is_some())
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Binds every ported reach end to its port flux for the attempt of `stage`.
    ///
    /// Each ported reach reserves its accepted-start operands under its flux port endpoints.
    /// Without ports the binding is empty and every reach keeps its start reservation.
    pub fn bind_ports(
        &self,
        stage: &SurfaceCouplingStage<R>,
        start: &Start<R>,
        reaches: &[ChannelReach<R>],
        reach_ids: &[&str],
    ) -> Result<PortBinding<R>, SurfaceChannelError> {
        let mut binding = PortBinding {
            reservations: (0..reaches.len()).map(|_| None).collect(),
            limit_s: f64::INFINITY,
            fluxes: Vec::new(),
            surface_rate_m3_per_s: [0.0; 2],
            boundaries: Vec::new(),
        };
        let (Some(topology), Some(ports)) = (&self.ports, &start.ports) else {
            return Ok(binding);
        };
        let fluxes = stage.port_fluxes().ok_or_else(|| {
            SurfaceChannelError::Numerical("a ported surface stage lacks its port fluxes".into())
        })?;
        let host = HostFluxes::read(&self.client, fluxes)?;
        let mut first = 0;
        for (index, (port, &end)) in topology.ports.iter().zip(&ports.ends).enumerate() {
            binding
                .fluxes
                .push(port_flux(&host, index, port, end, first)?);
            first += port.faces.len();
        }
        binding.surface_rate_m3_per_s = [fluxes.inflow_rate_m3_per_s, fluxes.outflow_rate_m3_per_s];
        let device = self.client.device();
        let scalar = |value: f64| Tensor::<R>::from_slice(&[value], &[1], device);
        let mut boundaries = Vec::with_capacity(binding.fluxes.len());
        for (port, flux) in topology.ports.iter().zip(&binding.fluxes) {
            let reach = &reaches[port.reach_index];
            if rest_wall(reach, port, flux).map_err(|source| SurfaceChannelError::Channel {
                reach_id: reach_ids[port.reach_index].to_string(),
                source,
            })? {
                boundaries.push(ChannelTensorBoundary::Wall);
                continue;
            }
            let tensor = ChannelTensorFluxPort::new(
                scalar(flux.outward_water_rate_m3_per_s)?,
                scalar(flux.outward_momentum_rate_m4_per_s2)?,
                scalar(flux.signal_speed_m_per_s)?,
                flux.exterior_invariant_m_per_s.map(scalar).transpose()?,
                PORT_PROVENANCE,
            )
            .map_err(|source| SurfaceChannelError::Channel {
                reach_id: reach_ids[port.reach_index].to_string(),
                source,
            })?;
            boundaries.push(ChannelTensorBoundary::FluxPort(Box::new(tensor)));
        }
        let zeros = Tensor::<R>::zeros(&[self.link_count], DType::F64, device)?;
        for (reach, ends) in topology.reach_ports.iter().enumerate() {
            // A pool-ended reach reserves per evaluation under its reservoir heads.
            if ends.iter().all(Option::is_none) || self.is_pool_ended(reach) {
                continue;
            }
            let channel = |source| SurfaceChannelError::Channel {
                reach_id: reach_ids[reach].to_string(),
                source,
            };
            let bound = &start.forcing.reaches[reach];
            let policy = |end: ChannelEnd| {
                let caller = match end {
                    ChannelEnd::Left => &bound.left,
                    ChannelEnd::Right => &bound.right,
                };
                ends[end_slot(end)]
                    .and_then(|port| boundaries.get(port))
                    .or(caller.as_ref())
                    .ok_or_else(|| {
                        SurfaceChannelError::Numerical(
                            "a reach end lacks its caller policy and its port".into(),
                        )
                    })
            };
            let left = policy(ChannelEnd::Left)?;
            let right = policy(ChannelEnd::Right)?;
            let mut inputs = bound.lateral.clone();
            if let Some(linked) = self.linked.iter().position(|entry| entry.reach == reach) {
                inputs.extend(self.exchange_inputs(linked, &zeros, &zeros, &zeros)?);
            }
            let child = &reaches[reach];
            let forcing = child
                .bind_coupling_tensor(start.time, stage.end(), Some(left), Some(right), &inputs)
                .map_err(channel)?;
            let reservation = child.coupling_reservation(&forcing).map_err(channel)?;
            binding.limit_s = binding
                .limit_s
                .min(reservation.start_cfl_s)
                .min(reservation.face_depletion_limit_s);
            binding.reservations[reach] = Some(reservation);
        }
        binding.boundaries = boundaries;
        Ok(binding)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::SurfaceChannelPortTransfer;
    use super::super::fixture::{BETA, TestResult, x_face};
    use super::super::scenario::{
        Batch, CELLS, GRAVITY, Reach, Scenario, east, receipt, step, still, west,
    };
    use super::raised;
    use crate::coupling::SurfaceChannelPortRegime;
    use crate::solver::BoundaryFace;
    use hydr_core::model::CellState;

    /// `alpha` in state `alpha` against `surface` through the east port, on a flat 1 m bed.
    fn east_case(alpha: Reach, surface: Vec<CellState>) -> Scenario {
        Scenario {
            surface_bed: 1.0,
            surface,
            alpha,
            beta: Reach::rest(0.25, 1.5),
            link: false,
            ports: vec![east("p")],
            surveyed: false,
            open: Vec::new(),
            ends: Vec::new(),
        }
    }

    /// One interval toward `dt`, with the batch and its port receipt.
    fn east_step(case: &Scenario, dt: f64) -> TestResult<(Batch, SurfaceChannelPortTransfer)> {
        let batch = step(&case.build()?, dt)?;
        let transfer = receipt(&batch)?.transfers[0].clone();
        Ok((batch, transfer))
    }

    #[test]
    fn the_port_speed_covers_the_reach_end_across_a_strong_shock() -> TestResult {
        // A reach end at 10 m/s runs into still water of the same depth. The shock bounds of
        // the HLL face fall below the end cell's |u| + c, so the reach end sets the speed.
        let case = east_case(Reach::uniform(1.0, 1.0, 10.0), vec![still(1.0); CELLS]);
        let (batch, transfer) = east_step(&case, 1.0)?;
        let characteristic = 10.0 + GRAVITY.sqrt();
        assert_eq!(
            transfer.signal_speed_m_per_s.to_bits(),
            raised(characteristic).to_bits()
        );
        let surface = batch.surface().last_step()?.ok_or("a surface receipt")?;
        let port_faces: Vec<BoundaryFace> = vec![x_face(4, 0), x_face(4, 1)];
        let hll = surface
            .faces
            .iter()
            .filter(|face| port_faces.contains(&face.face))
            .map(|face| face.signal_speed)
            .fold(0.0, f64::max);
        assert!(hll > 0.0 && hll < characteristic, "HLL bound={hll}");
        // The ported reach's CFL limit shortened the requested second.
        let diagnostics = batch.last_step()?.ok_or("a coordinator receipt")?;
        assert!(diagnostics.retries >= 1);
        let reach = batch.reach(super::super::fixture::ALPHA).ok_or("alpha")?;
        let limits = reach.last_step()?.ok_or("a reach receipt")?;
        assert!(batch.time() <= limits.cfl_limit_s);
        // An unported reach keeps its caller endpoints.
        assert!(batch.reach(BETA).is_some());
        Ok(())
    }

    /// A still port hands the rest end cell of a surveyed rectangle its own wall face.
    ///
    /// The reach rests at stage 0.8 m on a 0.5 m bed, below the dry surface bed at 1 m, so the
    /// port moves no water. The end cell's two-part rest pressure has a nonzero low part.
    ///
    /// Frozen criteria, for the surveyed rectangle over 50 intervals of 0.1 s:
    /// - every port transfer has the `Dry` regime and a `+0.0` signed volume
    /// - the ported end face carries both momentum parts of the next interior face
    /// - every reach cell stays bit-identical to its start
    ///
    /// The rectangular reach of the same state keeps its flux port and the same rest bits.
    #[test]
    fn a_still_port_hands_a_surveyed_rest_end_its_wall_face() -> TestResult {
        for surveyed in [true, false] {
            let mut case = Scenario::rest(1.0, 0.8, 0.5, vec![west("p")]);
            case.surveyed = surveyed;
            case.link = false;
            let mut batch = case.build()?;
            let start = batch
                .reach(super::super::fixture::ALPHA)
                .ok_or("alpha")?
                .state()?;
            for _ in 0..50 {
                batch = step(&batch, 0.1)?;
                let transfer = &receipt(&batch)?.transfers[0];
                assert_eq!(transfer.regime, SurfaceChannelPortRegime::Dry);
                assert_eq!(transfer.signed_volume_m3.to_bits(), 0.0f64.to_bits());
                let reach = batch.reach(super::super::fixture::ALPHA).ok_or("alpha")?;
                let last = reach.last_step()?.ok_or("a reach receipt")?;
                let parts = |face: usize| {
                    [
                        last.face_momentum_m4_per_s[face].to_bits(),
                        last.face_momentum_low_m4_per_s[face].to_bits(),
                    ]
                };
                if surveyed {
                    assert_ne!(last.face_momentum_low_m4_per_s[1], 0.0);
                }
                assert_eq!(parts(0), parts(1), "surveyed={surveyed}");
                assert_eq!(reach.state()?, start, "surveyed={surveyed}");
            }
        }
        Ok(())
    }

    #[test]
    fn raising_adds_four_epsilon_and_a_quantum() {
        assert!(raised(1.0) > 1.0 && raised(1.0) <= 1.0 + 5.0 * f64::EPSILON);
        assert!(raised(-1.0) > -1.0);
        assert!(raised(0.0) > 0.0);
    }
}

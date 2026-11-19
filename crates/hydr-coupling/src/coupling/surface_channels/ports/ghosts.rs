//! Accepted-start ghost states of every port, built once per common trial.

use super::super::SurfaceChannelError;
use super::super::context::{Context, Ops};
use crate::channel::ChannelEnd;
use crate::channel::ChannelReach;
use crate::solver::Solver;
use hydr_solver::internal::PortGhosts;
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Accepted-start state of one ported reach end cell, on the host.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in super::super) struct PortEnd {
    /// Depth `A / b`, in m.
    pub depth_m: f64,
    /// Velocity along the reach's outward normal, in m/s. Zero when dry.
    pub outward_velocity_m_per_s: f64,
}

/// Ghost states and host end states of every port, from the accepted reaches.
pub(in super::super) struct PortStart<R: Runtime<DType = DType>> {
    pub ghosts: PortGhosts<R>,
    /// End state of every port, in port order.
    pub ends: Vec<PortEnd>,
}

/// Outward normal sign of a reach end: `-1` left, `+1` right.
pub(in super::super) fn outward(end: ChannelEnd) -> f64 {
    match end {
        ChannelEnd::Left => -1.0,
        ChannelEnd::Right => 1.0,
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Ghost states of every port from the accepted `reaches`, or `None` without ports.
    ///
    /// Each ghost holds its reach end cell's depth `A / b` and discharge per width `Q / b`.
    pub fn port_start(
        &self,
        surface: &Solver<R>,
        reaches: &[ChannelReach<R>],
        reach_ids: &[&str],
    ) -> Result<Option<PortStart<R>>, SurfaceChannelError> {
        let Some(topology) = &self.ports else {
            return Ok(None);
        };
        let c = &self.client;
        let mut depth = Vec::with_capacity(topology.ports.len());
        let mut discharge = Vec::with_capacity(topology.ports.len());
        let mut area = Vec::with_capacity(topology.ports.len());
        let mut flow = Vec::with_capacity(topology.ports.len());
        for port in &topology.ports {
            let state = reaches[port.reach_index]
                .coupling_end_state(port.end)
                .map_err(|source| SurfaceChannelError::Channel {
                    reach_id: reach_ids[port.reach_index].to_string(),
                    source,
                })?;
            depth.push(state.depth_m);
            discharge.push(state.discharge_per_width_m2_per_s);
            area.push(state.area_m2);
            flow.push(state.discharge_m3_per_s);
        }
        let join = |parts: &[Tensor<R>]| c.cat(&parts.iter().collect::<Vec<_>>(), 0);
        let depth = join(&depth)?;
        let discharge = join(&discharge)?;
        let ghosts = surface.coupling_port_ghosts(&topology.surface, &depth, &discharge)?;
        let count = topology.ports.len();
        let host = c
            .cat(&[&depth, &join(&area)?, &join(&flow)?], 0)?
            .try_to_vec::<f64>()?;
        let ends = topology
            .ports
            .iter()
            .enumerate()
            .map(|(index, port)| {
                let read = |offset: usize| {
                    host.get(offset * count + index).copied().ok_or_else(|| {
                        SurfaceChannelError::Numerical("port end read lacks a value".into())
                    })
                };
                let (depth_m, area_m2, discharge) = (read(0)?, read(1)?, read(2)?);
                let velocity = if area_m2 > 0.0 {
                    discharge / area_m2
                } else {
                    0.0
                };
                Ok(PortEnd {
                    depth_m,
                    outward_velocity_m_per_s: outward(port.end) * velocity,
                })
            })
            .collect::<Result<_, SurfaceChannelError>>()?;
        Ok(Some(PortStart { ghosts, ends }))
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{ALPHA, BETA, REACH_WIDTH, TestResult, x_face};
    use super::super::scenario::{
        Batch, CELLS, GRAVITY, Reach, Scenario, east, receipt, step, west,
    };
    use crate::channel::{ChannelBoundary, ChannelCellResult, ChannelEnd};
    use crate::coupling::SurfaceChannelPortRegime;
    use crate::solver::{BoundaryForcing, ExteriorBoundary, ExteriorState};
    use hydr_core::model::CellState;

    /// Requested duration of every interval, in s. The CFL limit shortens it.
    const DT: f64 = 0.1;

    /// Accepted intervals of every run.
    const INTERVALS: usize = 50;

    /// Frozen bound on every `|q|`, `|Q|`, and stage change at rest.
    const REST: f64 = 1e-12;

    /// Frozen bound on the relative change of every `h` and `q` in uniform flow.
    const UNIFORM: f64 = 1e-12;

    /// Surface bed of every case, in m.
    const SURFACE_BED: f64 = 1.0;

    /// Depth of the uniform state, in m.
    const DEPTH: f64 = 0.5;

    fn reach_cells(batch: &Batch, id: &str) -> TestResult<Vec<ChannelCellResult>> {
        Ok(batch.reach(id).ok_or("the reach exists")?.state()?)
    }

    /// A surveyed rectangle `alpha` of the face width at rest, joined to a surface at rest.
    ///
    /// Frozen criteria, the same as the rectangular rest case of `ports_rest.rs`, fixed before
    /// the first run. The reach bed equals the surface bed, lies below it, and lies above it.
    /// The port covers the two west faces, so the face width is the section width.
    /// - Every interval has the `Subcritical` regime and pairs within its bound.
    /// - After `INTERVALS` accepted intervals every `|q|`, `|Q|` and stage change of every wet
    ///   cell stays within `REST` of rest at stage 1.6 m.
    /// - The lifetime composite balance holds.
    #[test]
    fn a_surveyed_rectangle_port_holds_a_lake_at_rest() -> TestResult {
        let stage = 1.6;
        for alpha_bed in [SURFACE_BED, 0.5, 1.125] {
            let mut case = Scenario::rest(SURFACE_BED, stage, alpha_bed, vec![west("p")]);
            case.surveyed = true;
            let mut batch = case.build()?;
            for _ in 0..INTERVALS {
                batch = step(&batch, DT)?;
                let transfer = &receipt(&batch)?.transfers[0];
                assert_eq!(
                    transfer.regime,
                    SurfaceChannelPortRegime::Subcritical,
                    "bed {alpha_bed}"
                );
                assert!(transfer.paired_gap_m3.abs() <= transfer.paired_bound_m3);
            }
            assert_eq!(batch.accepted_steps(), INTERVALS, "bed {alpha_bed}");
            for cell in batch.surface().state()?.cells() {
                assert!(cell.qx.abs() <= REST && cell.qy.abs() <= REST, "{cell:?}");
                assert!((cell.h + SURFACE_BED - stage).abs() <= REST, "{cell:?}");
            }
            for id in [ALPHA, BETA] {
                for cell in reach_cells(&batch, id)? {
                    assert!(cell.discharge_m3_per_s.abs() <= REST, "{id}: {cell:?}");
                    let wet = cell.stage_m.ok_or("every reach cell is wet")?;
                    assert!((wet - stage).abs() <= REST, "{id}: {cell:?}");
                }
            }
            let balance = batch.ledger()?;
            assert!(balance.residual_m3.abs() <= balance.bound_m3, "{balance:?}");
        }
        Ok(())
    }

    /// Uniform flow at Froude number `froude` through a surveyed rectangle `alpha` that ends in
    /// the east port. `toward_surface` runs the reach into the surface.
    fn uniform(froude: f64, toward_surface: bool) -> Scenario {
        let speed = froude * (GRAVITY * DEPTH).sqrt();
        let velocity = if toward_surface { speed } else { -speed };
        let q = DEPTH * velocity;
        let mut case = Scenario::rest(
            SURFACE_BED,
            SURFACE_BED + DEPTH,
            SURFACE_BED,
            vec![east("p")],
        );
        case.surveyed = true;
        case.link = false;
        case.alpha = Reach::uniform(SURFACE_BED, DEPTH, q);
        case.surface = vec![
            CellState {
                h: DEPTH,
                qx: -q,
                qy: 0.0,
            };
            CELLS
        ];
        case.open = (0..2)
            .map(|row| {
                BoundaryForcing::Exterior(ExteriorBoundary {
                    face: x_face(0, row),
                    state: ExteriorState::Wet {
                        elevation: SURFACE_BED + DEPTH,
                        outward_velocity: velocity,
                        tangential_velocity: 0.0,
                    },
                })
            })
            .collect();
        case.ends = vec![(
            ALPHA,
            ChannelEnd::Left,
            ChannelBoundary::Exterior {
                depth_m: DEPTH,
                velocity_m_per_s: velocity,
            },
        )];
        case
    }

    /// `|actual - expected|` over `|scale|`.
    fn change(actual: f64, expected: f64, scale: f64) -> f64 {
        (actual - expected).abs() / scale.abs()
    }

    /// Uniform flow through a surveyed rectangle port.
    ///
    /// Frozen criteria, the same as `ports_uniform.rs`, fixed before the first run. Both
    /// children hold one frictionless state on one flat bed. For each of `INTERVALS` accepted
    /// intervals:
    /// - The port has the stated regime.
    /// - The port volume and the reach momentum impulse each match the uniform state's flux
    ///   within `UNIFORM` relative.
    /// - The port pairs within its bound.
    ///
    /// After the run every `h`, `q`, depth and discharge changes by at most `UNIFORM` relative.
    fn holds(froude: f64, toward_surface: bool, regime: SurfaceChannelPortRegime) -> TestResult {
        let name = format!("Fr {froude}, toward surface {toward_surface}");
        let case = uniform(froude, toward_surface);
        let mut batch = case.build()?;
        let surface = batch.surface().state()?.cells().to_vec();
        let reach = reach_cells(&batch, ALPHA)?;
        let q = surface[0].qx.abs();
        let discharge = REACH_WIDTH * q;
        let momentum = REACH_WIDTH * (q * q / DEPTH + 0.5 * GRAVITY * DEPTH * DEPTH);
        for _ in 0..INTERVALS {
            batch = case.step(&batch, DT)?;
            let step = batch.last_step()?.ok_or("an accepted interval")?;
            let transfer = &step.ports[0];
            assert_eq!(transfer.regime, regime, "{name}");
            let dt = step.end_s - step.start_s;
            let signed = if toward_surface {
                -discharge
            } else {
                discharge
            };
            assert!(
                change(transfer.signed_volume_m3, signed * dt, signed * dt) <= UNIFORM,
                "{name}: {transfer:?}"
            );
            assert!(
                change(
                    transfer.reach_momentum_impulse_m4_per_s,
                    momentum * dt,
                    momentum * dt
                ) <= UNIFORM,
                "{name}: {transfer:?}"
            );
            assert!(transfer.paired_gap_m3.abs() <= transfer.paired_bound_m3);
        }
        assert_eq!(batch.accepted_steps(), INTERVALS);
        for (cell, start) in batch.surface().state()?.cells().iter().zip(&surface) {
            assert!(
                change(cell.h, start.h, start.h) <= UNIFORM,
                "{name}: {cell:?}"
            );
            assert!(change(cell.qx, start.qx, q) <= UNIFORM, "{name}: {cell:?}");
            assert!(change(cell.qy, start.qy, q) <= UNIFORM, "{name}: {cell:?}");
        }
        for (cell, start) in reach_cells(&batch, ALPHA)?.iter().zip(&reach) {
            assert!(
                change(cell.depth_m, start.depth_m, start.depth_m) <= UNIFORM,
                "{name}: {cell:?}"
            );
            assert!(
                change(
                    cell.discharge_m3_per_s,
                    start.discharge_m3_per_s,
                    start.discharge_m3_per_s
                ) <= UNIFORM,
                "{name}: {cell:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn subcritical_uniform_flow_crosses_a_surveyed_rectangle_port_unchanged() -> TestResult {
        holds(0.5, true, SurfaceChannelPortRegime::Subcritical)?;
        holds(0.5, false, SurfaceChannelPortRegime::Subcritical)
    }

    #[test]
    fn supercritical_uniform_flow_crosses_a_surveyed_rectangle_port_unchanged() -> TestResult {
        holds(2.0, true, SurfaceChannelPortRegime::SupercriticalToSurface)?;
        holds(2.0, false, SurfaceChannelPortRegime::SupercriticalToReach)
    }

    /// A rectangular and a surveyed-rectangle `alpha` with identical start states give equal
    /// port receipts.
    ///
    /// Frozen criterion, fixed before the first run: after one interval of `DT / 10`, both
    /// batches accept the same interval and their `PortInterval` receipts are equal field for
    /// field, with every `f64` equal bit for bit. The reach stands above the surface and moves.
    #[test]
    fn a_surveyed_rectangle_gives_the_rectangular_port_receipt() -> TestResult {
        let run = |surveyed: bool| -> TestResult<(Batch, f64)> {
            let mut case = Scenario::rest(SURFACE_BED, 1.6, 0.5, vec![west("p")]);
            case.surveyed = surveyed;
            case.link = false;
            case.alpha = Reach {
                bed: 0.5,
                depth: [1.3, 1.25, 1.2],
                discharge_per_width: [0.2, 0.15, 0.1],
            };
            let batch = step(&case.build()?, DT / 10.0)?;
            let end = batch.last_step()?.ok_or("an accepted interval")?.end_s;
            Ok((batch, end))
        };
        let (rectangular, rectangular_end) = run(false)?;
        let (surveyed, surveyed_end) = run(true)?;
        assert_eq!(rectangular_end.to_bits(), surveyed_end.to_bits());
        let (a, b) = (receipt(&rectangular)?, receipt(&surveyed)?);
        assert_eq!(a, b);
        let (a, b) = (&a.transfers[0], &b.transfers[0]);
        for (x, y) in [
            (a.signed_volume_m3, b.signed_volume_m3),
            (
                a.reach_momentum_impulse_m4_per_s,
                b.reach_momentum_impulse_m4_per_s,
            ),
            (a.signal_speed_m_per_s, b.signal_speed_m_per_s),
            (a.paired_gap_m3, b.paired_gap_m3),
            (a.paired_bound_m3, b.paired_bound_m3),
        ] {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        assert_ne!(a.signed_volume_m3, 0.0);
        Ok(())
    }
}

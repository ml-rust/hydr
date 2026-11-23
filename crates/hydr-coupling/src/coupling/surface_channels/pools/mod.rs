//! Level pools as coordinator nodes: declarations, admission, and the pool child.
//!
//! One [`crate::storage::LevelPoolBatch`] holds every pool in ascending identity. Pools follow
//! the surface lanes and channel slots in the donor groups `S + C + P`. Each pool offers its
//! post-source water: local rain, then capped bottom loss and evaporation, before any withdrawal.
//! Incoming link water and external inflows fund no withdrawal in the same interval.

mod admission;
mod capacity;
mod child;
mod fns;
mod topology;

#[cfg(test)]
pub(super) mod scenario;

pub(super) use admission::{check_attached, check_pools};
pub(super) use child::{
    PoolAttempt, PoolChild, PoolCompletion, PoolStart, charge, inflow_error, pool_error,
};
pub(super) use fns::PoolParts;
pub(super) use topology::{LinkSides, PoolCurveState, PoolTopology};

use super::SurfaceChannelSetup;
use super::pool_ends::SurfaceChannelPoolEnd;
use crate::coupling::level_pools::{CoupledPoolInflowBinding, CoupledPoolSupport};
use crate::storage::{LevelPoolConfig, LevelPoolElement, LevelPoolSourcePolicy};

/// The pool declarations of one setup: pools, support, sources, inflows, controls, and ends.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct SurfaceChannelPools {
    /// Pool definitions in setup order. Their curves use the surface vertical datum.
    pub pools: Vec<LevelPoolElement>,
    /// One support declaration per pool, in the order of `pools`.
    pub support: Vec<CoupledPoolSupport>,
    /// One source policy per pool, in the order of `pools`, or `None` without local sources.
    pub sources: Option<Vec<LevelPoolSourcePolicy>>,
    /// External inflow bindings into the pools.
    pub inflows: Vec<CoupledPoolInflowBinding>,
    /// Dimensional controls of the pool child.
    pub storage: LevelPoolConfig,
    /// Reach ends joined to the pools in setup order.
    pub ends: Vec<SurfaceChannelPoolEnd>,
}

impl SurfaceChannelPools {
    /// The pool declarations of `setup`.
    pub fn of(setup: &SurfaceChannelSetup) -> Self {
        Self {
            pools: setup.pools.clone(),
            support: setup.pool_support.clone(),
            sources: setup.pool_sources.clone(),
            inflows: setup.pool_inflows.clone(),
            storage: setup.storage.clone(),
            ends: setup.pool_ends.clone(),
        }
    }
}

/// Admitted pools in ascending identity.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct CheckedPools {
    pub declared: SurfaceChannelPools,
    /// Setup index of each pool in ascending identity.
    pub order: Vec<usize>,
    /// Pool identities in ascending order.
    pub ids: Vec<String>,
}

impl CheckedPools {
    /// Identity position and definition of pool `id`.
    pub fn find(&self, id: &str) -> Option<(usize, &LevelPoolElement)> {
        let position = self
            .ids
            .binary_search_by(|probe| probe.as_str().cmp(id))
            .ok()?;
        let setup = *self.order.get(position)?;
        Some((position, self.declared.pools.get(setup)?))
    }

    /// Pool definitions in ascending identity.
    pub fn elements(&self) -> Vec<LevelPoolElement> {
        self.order
            .iter()
            .map(|&index| self.declared.pools[index].clone())
            .collect()
    }

    /// `values` in setup pool order, reordered into ascending identity.
    pub fn ordered<T: Clone>(&self, values: &[T]) -> Vec<T> {
        self.order
            .iter()
            .map(|&index| values[index].clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::scenario::{
        FREE_WEIR, PoolForcing, advance, build, channel, check, declaration, inflow_binding,
        inflow_input, link, long_weir, pond, pool, stages, surface, surface_stage, volumes,
    };
    use crate::coupling::surface_channels::structures::scenario::{
        AREA, EQUAL_HEADS, PUMP_CURVE, Tanks, TestResult, allowance, gate_law, orifice_law,
        pump_curve, pump_law, receipt, weir_law,
    };
    use crate::structures::{DirectionPolicy, StructureOperatingPoint};

    /// Requested duration of every interval, in s.
    const DT: f64 = 0.5;

    fn passive(
        id: &str,
        law: crate::structures::StructureLaw,
        operation: StructureOperatingPoint,
        a: (super::super::structures::SurfaceChannelNode, f64),
        b: (super::super::structures::SurfaceChannelNode, f64),
    ) -> super::super::structures::SurfaceChannelStructure {
        link(id, law, DirectionPolicy::Bidirectional, operation, a, b)
    }

    /// A pond level with the surface cell through a submerged weir, an orifice, and an open gate.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Over 50 intervals every link volume is `0.0` bit for bit and every regime is equal heads.
    /// - The pond and surface stages each move by at most 1e-12 m.
    /// - Every interval passes its mass bound, every pairing, and the composite balance.
    #[test]
    fn a_pool_at_rest_stays_at_rest() -> TestResult {
        let structures = vec![
            passive(
                "weir",
                weir_law(0.4)?,
                StructureOperatingPoint::Passive,
                (surface(), 0.0),
                (pool("pond"), 0.0),
            ),
            passive(
                "culvert",
                orifice_law(0.1, 0.3)?,
                StructureOperatingPoint::Passive,
                (surface(), 0.0),
                (pool("pond"), 0.0),
            ),
            passive(
                "gate",
                gate_law(0.2)?,
                StructureOperatingPoint::GateOpening { opening_m: 0.2 },
                (pool("pond"), 0.0),
                (surface(), 0.0),
            ),
        ];
        let pools = declaration(vec![pond("pond", 0.0, AREA, 2.0)?], false, Vec::new());
        let mut batch = build(Tanks::at(0.6, 0.0), pools, &[0.6 * AREA], structures)?;
        let start = [surface_stage(&batch)?, stages(&batch)?[0]];
        let none = PoolForcing::none();
        for _ in 0..50 {
            batch = advance(&batch, DT, &none)?;
            check(&batch, &[0.0])?;
            for id in ["weir", "culvert", "gate"] {
                let link = receipt(&batch, id)?;
                assert_eq!(link.signed.to_bits(), 0.0f64.to_bits(), "{id}: {link:?}");
                assert_eq!(link.regime, EQUAL_HEADS, "{id}: {link:?}");
            }
        }
        let end = [surface_stage(&batch)?, stages(&batch)?[0]];
        assert!((end[0] - start[0]).abs() <= 1e-12, "{start:?} {end:?}");
        assert!((end[1] - start[1]).abs() <= 1e-12, "{start:?} {end:?}");
        Ok(())
    }

    /// A free weir drains a constant-area pond onto a low dry surface cell.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every interval passes its mass bound, every pairing, and the composite balance.
    /// - Every accepted volume is nonnegative, and the first interval flows free.
    /// - After 20 intervals the head above the crest matches
    ///   `h(t) = (h0^-1/2 + C L t / (2A))^-2` within 1e-2 relative.
    /// - The pond never ends below its crest by more than the head allowance.
    /// - The pond loses exactly what the surface gains, within 1e-9 m³.
    #[test]
    fn a_free_weir_drains_a_pool_onto_the_surface() -> TestResult {
        let crest = 0.5;
        let structures = vec![link(
            "spill",
            weir_law(crest)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::Passive,
            (pool("pond"), 0.0),
            (surface(), -2.0),
        )];
        let pools = declaration(vec![pond("pond", 0.0, AREA, 2.0)?], false, Vec::new());
        let tanks = Tanks {
            surface_bed: -2.0,
            surface_stage: -2.0,
            reach_bed: 0.0,
            reach_stage: 0.0,
        };
        let mut batch = build(tanks, pools, &[AREA], structures)?;
        let none = PoolForcing::none();
        let mut moved = 0.0;
        let mut elapsed = 0.0;
        for index in 0..20 {
            batch = advance(&batch, DT, &none)?;
            check(&batch, &[0.0])?;
            let spill = receipt(&batch, "spill")?;
            assert!(spill.signed >= 0.0, "{spill:?}");
            if index == 0 {
                assert_eq!(spill.regime, FREE_WEIR, "{spill:?}");
            }
            moved += spill.signed;
            let interval = batch.last_step()?.ok_or("an accepted interval")?;
            elapsed += interval.end_s - interval.start_s;
            let stage = stages(&batch)?[0];
            assert!(stage >= crest - allowance([stage, crest]), "stage={stage}");
        }
        let h0 = 0.5f64;
        let exact = (h0.powf(-0.5) + 1.7 * elapsed / (2.0 * AREA)).powi(-2);
        let head = stages(&batch)?[0] - crest;
        assert!(
            (head - exact).abs() <= 1e-2 * exact,
            "h={head}, exact={exact}"
        );
        let lost = AREA - volumes(&batch)?[0];
        let gained = (surface_stage(&batch)? + 2.0) * AREA;
        assert!((lost - moved).abs() <= 1e-9, "lost={lost}, moved={moved}");
        assert!(
            (gained - moved).abs() <= 1e-9,
            "gained={gained}, moved={moved}"
        );
        Ok(())
    }

    /// An orifice between two ponds until their heads level.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every interval passes its mass bound, every pairing, and the composite balance.
    /// - Every accepted volume is nonnegative: no overshoot.
    /// - Within 200 intervals the link reaches equal heads and carries `0.0`.
    /// - The final heads differ by at most twice the head allowance.
    /// - The moved volume equals half the initial head difference times the area within 1e-9 m³.
    #[test]
    fn an_orifice_levels_two_pools() -> TestResult {
        let structures = vec![passive(
            "culvert",
            orifice_law(0.1, 0.3)?,
            StructureOperatingPoint::Passive,
            (pool("high"), 0.0),
            (pool("low"), 0.0),
        )];
        let ponds = vec![pond("high", 0.0, AREA, 2.0)?, pond("low", 0.0, AREA, 2.0)?];
        let pools = declaration(ponds, false, Vec::new());
        let mut batch = build(Tanks::at(0.0, 0.0), pools, &[AREA, 0.5 * AREA], structures)?;
        let none = PoolForcing::none();
        let mut moved = 0.0;
        let mut levelled = false;
        for _ in 0..200 {
            batch = advance(&batch, DT, &none)?;
            check(&batch, &[0.0, 0.0])?;
            let culvert = receipt(&batch, "culvert")?;
            assert!(culvert.signed >= 0.0, "{culvert:?}");
            moved += culvert.signed;
            if culvert.regime == EQUAL_HEADS {
                assert_eq!(culvert.signed, 0.0, "{culvert:?}");
                levelled = true;
                break;
            }
        }
        assert!(levelled, "the orifice never reached equal heads");
        // Pools sort by identity: `high` then `low`.
        let end = stages(&batch)?;
        let end = [end[0], end[1]];
        assert!((end[0] - end[1]).abs() <= 2.0 * allowance(end), "{end:?}");
        let expected = 0.5 * AREA / 2.0;
        assert!((moved - expected).abs() <= 1e-9, "moved={moved}");
        Ok(())
    }

    /// A running pump lifts pond water into the reach cell.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every interval passes its mass bound, every pairing, and the composite balance.
    /// - Every one of 10 intervals keeps the curve regime and moves water from the pond.
    /// - The first interval moves `Q(midpoint lift) dt` within 1e-12 relative.
    #[test]
    fn a_pump_lifts_a_pool_into_a_channel_cell() -> TestResult {
        let structures = vec![link(
            "pump",
            pump_law(0.3, None)?,
            DirectionPolicy::AtoBOnly,
            StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            (pool("sump"), 0.0),
            (channel(), 0.0),
        )];
        let pools = declaration(vec![pond("sump", 0.0, AREA, 2.0)?], false, Vec::new());
        let mut batch = build(Tanks::at(0.0, 0.5), pools, &[AREA], structures)?;
        let none = PoolForcing::none();
        for index in 0..10 {
            batch = advance(&batch, DT, &none)?;
            check(&batch, &[0.0])?;
            let pump = receipt(&batch, "pump")?;
            assert_eq!(pump.regime, PUMP_CURVE, "{pump:?}");
            assert!(pump.signed > 0.0, "{pump:?}");
            if index == 0 {
                let interval = batch.last_step()?.ok_or("an accepted interval")?;
                let dt = interval.end_s - interval.start_s;
                let expected = pump_curve(pump.midpoint[1] - pump.midpoint[0]) * dt;
                assert!(
                    (pump.signed - expected).abs() <= 1e-12 * expected,
                    "{pump:?}, expected={expected}"
                );
            }
        }
        Ok(())
    }

    /// Rain and an external inflow fill a pond behind a weir crest it never reaches.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every interval passes its mass bound, every pairing with the external inflow, and the
    ///   composite balance.
    /// - The weir carries `0.0` in every interval.
    /// - After 10 intervals the pond holds its start volume plus rain on its area plus the
    ///   inflow, within 1e-9 m³.
    #[test]
    fn sources_and_an_inflow_fill_a_pool() -> TestResult {
        let structures = vec![link(
            "spill",
            weir_law(2.0)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::Passive,
            (pool("pond"), 0.0),
            (surface(), 0.0),
        )];
        let pools = declaration(
            vec![pond("pond", 0.0, AREA, 4.0)?],
            true,
            vec![inflow_binding("pipe", "pond")],
        );
        let mut batch = build(Tanks::at(0.0, 0.0), pools, &[0.5 * AREA], structures)?;
        let (rain, rate) = (1e-3, 0.01);
        let forcing = PoolForcing {
            sources: Some(vec![crate::storage::LevelPoolSources {
                rain_rate_m_per_s: rain,
                potential_evaporation_rate_m_per_s: 0.0,
                bottom_loss_rate_m3_per_s: 0.0,
            }]),
            inflows: vec![inflow_input("pipe", rate)],
        };
        let mut elapsed = 0.0;
        for _ in 0..10 {
            batch = advance(&batch, DT, &forcing)?;
            let interval = batch.last_step()?.ok_or("an accepted interval")?;
            let dt = interval.end_s - interval.start_s;
            elapsed += dt;
            check(&batch, &[rate * dt])?;
            assert_eq!(receipt(&batch, "spill")?.signed, 0.0);
        }
        let expected = 0.5 * AREA + rain * AREA * elapsed + rate * elapsed;
        let volume = volumes(&batch)?[0];
        assert!(
            (volume - expected).abs() <= 1e-9,
            "volume={volume}, expected={expected}"
        );
        Ok(())
    }

    /// One pond drains through three long weirs into a surface cell, a channel cell, and a
    /// second pond, with demand above its water. Rain falls on it and a pipe feeds it.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - The interval accepts and passes its mass bound, every pairing, and the composite balance.
    /// - Every weir's donor factor lies below one.
    /// - The pond keeps a nonnegative volume, at least the pipe volume of the interval.
    /// - The pond gives more than its start volume and at most its start volume plus its rain.
    #[test]
    fn donors_share_one_pool_across_three_links() -> TestResult {
        let ponds = vec![
            pond("pond", 0.0, AREA, 2.0)?,
            pond("sink", -5.0, AREA, 6.0)?,
        ];
        let pools = declaration(ponds, true, vec![inflow_binding("pipe", "pond")]);
        let weir = |id: &str, to| -> TestResult<_> {
            Ok(link(
                id,
                long_weir(0.0, 10.0)?,
                DirectionPolicy::Bidirectional,
                StructureOperatingPoint::Passive,
                (pool("pond"), 0.0),
                to,
            ))
        };
        let structures = vec![
            weir("to-surface", (surface(), -5.0))?,
            weir("to-channel", (channel(), -5.0))?,
            weir("to-sink", (pool("sink"), -5.0))?,
        ];
        let tanks = Tanks {
            surface_bed: -5.0,
            surface_stage: -5.0,
            reach_bed: -5.0,
            reach_stage: -5.0,
        };
        let start = 0.2 * AREA;
        let batch = build(tanks, pools, &[start, 0.0], structures)?;
        let (rain, rate, dt) = (0.01, 0.05, 1.0);
        let sources = crate::storage::LevelPoolSources {
            rain_rate_m_per_s: rain,
            potential_evaporation_rate_m_per_s: 0.0,
            bottom_loss_rate_m3_per_s: 0.0,
        };
        let forcing = PoolForcing {
            sources: Some(vec![sources, sources]),
            inflows: vec![inflow_input("pipe", rate)],
        };
        let batch = advance(&batch, dt, &forcing)?;
        let interval = batch.last_step()?.ok_or("an accepted interval")?;
        let dt = interval.end_s - interval.start_s;
        check(&batch, &[rate * dt, 0.0])?;
        let step = batch.last.as_ref().ok_or("an accepted interval")?;
        let factors = step.factor.try_to_vec::<f64>()?;
        assert!(factors.iter().all(|&f| f < 1.0), "{factors:?}");
        let given: f64 = step.signed_m3.try_to_vec::<f64>()?.iter().sum();
        let volume = volumes(&batch)?[0];
        assert!(volume >= rate * dt, "volume={volume}");
        assert!(given > start, "given={given}");
        assert!(given <= start + rain * AREA * dt, "given={given}");
        Ok(())
    }
}

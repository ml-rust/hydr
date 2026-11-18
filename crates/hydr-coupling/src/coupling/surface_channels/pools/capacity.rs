//! Pool capacity: candidates above the top curve knot.
//!
//! Water that no link exchange can remove rejects before iteration: post-source water plus
//! external inflow above a top knot. An iterate that overfills a pool takes saturated pool stages,
//! so the map stays continuous. Only a converged overflow rejects. Neither rejection retries.

use super::PoolChild;
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::context::{Context, Ops};
use crate::storage::LevelPoolSources;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ReduceOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Saturated pools of a candidate above a top curve knot.
pub(in super::super) struct PoolOverflow<R: Runtime<DType = DType>> {
    /// Curve stage of every pool, the top stage above the top knot, `[P]`.
    pub stage: Tensor<R>,
    /// One where a pool's candidate volume is positive, `[P]`.
    pub wet: Tensor<R>,
    /// The `pool-capacity-exceeded` rejection of this candidate.
    pub message: String,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Rejects a pool whose accepted water plus its source volumes and its `external` inflow,
    /// `[P]`, over `dt` lies above its top curve knot, before the pool child prepares its sources.
    ///
    /// Rain and evaporation act over the secant area of the accepted volume's segment, as the
    /// pool child applies them. The check uses nominal arithmetic.
    pub(super) fn check_source_capacity(
        &self,
        child: &PoolChild<R>,
        sources: &[LevelPoolSources],
        external: &Tensor<R>,
        dt: f64,
    ) -> Result<(), SurfaceChannelError> {
        let topology = self.pool_topology()?;
        let volumes = child.batch.volumes_tensor().try_to_vec::<f64>()?;
        let external = external.try_to_vec::<f64>()?;
        for (index, ((plan, source), (&volume, &inflow))) in topology
            .plans
            .iter()
            .zip(sources)
            .zip(volumes.iter().zip(&external))
            .enumerate()
        {
            let area = plan.secant_area(volume);
            let rain = source.rain_rate_m_per_s * area * dt;
            let losses = (source.potential_evaporation_rate_m_per_s * area
                + source.bottom_loss_rate_m3_per_s)
                * dt;
            let candidate = (volume + rain - losses).max(0.0) + inflow;
            let top = topology.maximum_m3.get(index).copied().unwrap_or(f64::NAN);
            if candidate > top {
                return Err(SurfaceChannelError::input(format!(
                    "pool-capacity-exceeded: pool={:?} candidate volume={candidate} before any link exchange exceeds the top curve knot volume={top}: declare a spill structure or extend the curve",
                    topology.ids.get(index).map_or("", String::as_str),
                )));
            }
        }
        Ok(())
    }

    /// The overflow of candidate volumes `available - outgoing + incoming` above the top curve
    /// knot, or `None` when every pool fits. The pool applies that sum in that order.
    ///
    /// Each pool stage saturates at its curve: a candidate above the top knot takes the top stage.
    /// `when` names the state in the rejection text.
    pub(super) fn overflow(
        &self,
        available: &Tensor<R>,
        outgoing: &Tensor<R>,
        incoming: &Tensor<R>,
        when: &str,
    ) -> Result<Option<PoolOverflow<R>>, SurfaceChannelError> {
        let c = &self.client;
        let topology = self.pool_topology()?;
        let candidate = c.add(&c.sub(available, outgoing)?, incoming)?;
        let over = c.gt(&candidate, &topology.maximum)?;
        if c.max(&over, &[0], false)?.item::<f64>()? == 0.0 {
            return Ok(None);
        }
        let candidate = candidate.try_to_vec::<f64>()?;
        let index = candidate
            .iter()
            .zip(&topology.maximum_m3)
            .position(|(volume, top)| volume > top)
            .unwrap_or(0);
        let message = format!(
            "pool-capacity-exceeded: pool={:?} candidate volume={} {when} exceeds the top curve knot volume={}: declare a spill structure or extend the curve",
            topology.ids.get(index).map_or("", String::as_str),
            candidate.get(index).copied().unwrap_or(f64::NAN),
            topology.maximum_m3.get(index).copied().unwrap_or(f64::NAN)
        );
        let stages: Vec<f64> = topology
            .plans
            .iter()
            .zip(&candidate)
            .map(|(plan, &volume)| plan.stage(volume))
            .collect();
        let wet: Vec<f64> = candidate
            .iter()
            .map(|&volume| f64::from(u8::from(volume > 0.0)))
            .collect();
        let device = c.device();
        Ok(Some(PoolOverflow {
            stage: Tensor::<R>::from_slice(&stages, &[stages.len()], device)?,
            wet: Tensor::<R>::from_slice(&wet, &[wet.len()], device)?,
            message,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::super::scenario::{
        PoolForcing, advance, build, check, declaration, link, pond, pool, steep_pump, surface,
        volumes,
    };
    use crate::coupling::surface_channels::SurfaceChannelError;
    use crate::coupling::surface_channels::structures::scenario::{
        AREA, Batch, DT, Tanks, TestResult, pump_law, weir_law,
    };
    use crate::structures::{DirectionPolicy, StructureOperatingPoint};

    /// Checks that `result` is a final `pool-capacity-exceeded` rejection of `pond` at `when`.
    fn capacity_rejection(result: Result<Batch, SurfaceChannelError>, when: &str) -> TestResult {
        match result {
            Err(error @ SurfaceChannelError::Input(_)) => {
                assert!(!error.is_retryable(), "{error}");
                let message = error.to_string();
                assert!(message.contains("pool-capacity-exceeded"), "{message}");
                assert!(message.contains("\"pond\""), "{message}");
                assert!(message.contains(when), "{message}");
                assert!(message.contains("declare a spill structure"), "{message}");
                Ok(())
            }
            Err(other) => Err(format!("expected an input rejection, got {other}").into()),
            Ok(_) => Err("the trial admitted a pool above its top knot".into()),
        }
    }

    /// A steep pump lifts a large pond into a small tank. At the start heads one interval of
    /// pumping would fill the tank past its top knot; at the midpoint heads it fits.
    ///
    /// With donor area 4, tank area 1, `Q = 0.5 (1 - lift)` and start stages 1.5 and 0.6, the
    /// midpoint rule moves `V = 0.95 dt / (1 + 0.3125 dt)`. At `dt = 0.45` the first iterate
    /// proposes `0.95 dt = 0.4275` m³ onto 0.6 m³ of water below a 1 m³ top knot.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - The interval accepts at the requested `dt` without a retry.
    /// - The first iterate's tank volume `0.6 + 0.95 dt` lies above the top knot.
    /// - The tank volume equals `0.6 + V` within 1e-9 m³.
    /// - The interval passes its mass bound, every pairing, and the composite balance.
    #[test]
    fn an_iterate_may_overfill_a_pool_that_converges_below_its_top() -> TestResult {
        let dt = 0.45;
        let structures = vec![link(
            "pump",
            steep_pump(0.3)?,
            DirectionPolicy::AtoBOnly,
            StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            (pool("source"), 0.0),
            (pool("tank"), 0.0),
        )];
        let ponds = vec![
            pond("source", 0.0, AREA, 2.0)?,
            pond("tank", 0.0, 1.0, 1.0)?,
        ];
        let pools = declaration(ponds, false, Vec::new());
        let batch = build(Tanks::at(0.0, 0.0), pools, &[1.5 * AREA, 0.6], structures)?;
        let batch = advance(&batch, dt, &PoolForcing::none())?;
        let interval = batch.last_step()?.ok_or("an accepted interval")?;
        assert_eq!(interval.retries, 0, "{interval:?}");
        assert_eq!((interval.end_s - interval.start_s).to_bits(), dt.to_bits());
        assert!(0.6 + 0.95 * dt > 1.0);
        check(&batch, &[0.0, 0.0])?;
        let expected = 0.6 + 0.95 * dt / (1.0 + 0.3125 * dt);
        let tank = volumes(&batch)?[1];
        assert!(
            (tank - expected).abs() <= 1e-9,
            "tank={tank}, expected={expected}"
        );
        Ok(())
    }

    /// A pump lifts surface water into a nearly full pond. Every interval it overfills the pond.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - The trial rejects at convergence with an input error that no shorter interval retries.
    /// - The message starts with `pool-capacity-exceeded`, names the pond, and names the fix.
    #[test]
    fn a_pool_that_converges_above_its_top_rejects_without_retry() -> TestResult {
        let structures = vec![link(
            "pump",
            pump_law(0.3, None)?,
            DirectionPolicy::AtoBOnly,
            StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            (surface(), 0.0),
            (pool("pond"), 0.0),
        )];
        let pools = declaration(vec![pond("pond", 0.0, AREA, 1.0)?], false, Vec::new());
        let batch = build(Tanks::at(1.0, 0.0), pools, &[3.98], structures)?;
        let result = advance(&batch, DT, &PoolForcing::none());
        capacity_rejection(result, "at the converged interval")
    }

    /// Rain alone fills a pond past its top knot.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - The trial rejects before iteration with an input error that no shorter interval
    ///   retries.
    /// - The message starts with `pool-capacity-exceeded`, names the pond, and names the fix.
    #[test]
    fn sources_above_capacity_reject_before_iteration() -> TestResult {
        let structures = vec![link(
            "spill",
            weir_law(2.0)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::Passive,
            (pool("pond"), 0.0),
            (surface(), 0.0),
        )];
        let pools = declaration(vec![pond("pond", 0.0, AREA, 1.0)?], true, Vec::new());
        let batch = build(Tanks::at(0.0, 0.0), pools, &[3.99], structures)?;
        let forcing = PoolForcing {
            sources: Some(vec![crate::storage::LevelPoolSources {
                rain_rate_m_per_s: 0.01,
                potential_evaporation_rate_m_per_s: 0.0,
                bottom_loss_rate_m3_per_s: 0.0,
            }]),
            inflows: Vec::new(),
        };
        capacity_rejection(advance(&batch, DT, &forcing), "before any link exchange")
    }
}

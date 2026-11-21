//! Pools and structures of the surface/channel coordinator on configured runtime clients.
//!
//! Frozen criterion, fixed before the first run: the case below runs on the CPU client and on
//! the CUDA client over the same intervals. After every accepted interval, the text of every
//! child state is equal between the two runs. Equal text means equal bits, signed zero
//! included. No difference is allowed, in any float. The text holds:
//! - the time and the accepted interval count
//! - the full checkpoint package: surface, reach, pool, operating packages, counters and the
//!   last interval with its diagnostics
//! - the coordinator, structure, pool-end, pool-inflow and link ledgers
//! - the live pool states, the surface state and the reach state
//! - the last interval receipt
//!
//! The CPU run must also cross a decision of each relay and the arrival of the actuated gate,
//! and the orifice, the pump, the gate and the pool end must each carry water.

pub(crate) use crate::pools_fixture;

use super::{Result, runtime};
use hydr::channel::{ChannelBoundary, ChannelEnd, ChannelLateralInput};
use hydr::coupling::{
    SurfaceChannelForcingInterval, SurfaceChannelOperating, SurfaceChannelOperationOwner,
    SurfaceChannelReachForcing, SurfaceChannelSetup,
};
use hydr::forcing::WaterAmount;
use hydr::structures::{ControlCommand, DirectionPolicy};
use pools_fixture::{
    Case, INFLOW, REACH, WITHDRAWAL, controlled_gate, controlled_pump, gate_law, inflow_binding,
    inflow_input, operating, orifice_law, pond, pool, pool_end, pump_law, signal, structure,
    surface,
};

/// Interval length, relay decision times, gate actuator rate and end of one run.
struct Timing {
    /// Requested duration of every interval, in s. Every decision time is a multiple of it.
    dt: f64,
    /// Decision times of both relays, in s.
    decisions: &'static [f64],
    /// Opening and closing rate of the gate actuator, in m/s.
    gate_rate: f64,
    /// End of the run, in s. It falls after the gate arrives.
    end: f64,
}

/// CPU run: the first decision falls at 1 s and the gate arrives at 4 s.
const LONG: Timing = Timing {
    dt: 0.5,
    decisions: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    gate_rate: 0.1,
    end: 8.0,
};

/// CPU/CUDA parity run: the first decision falls at 0.25 s and the gate arrives at 0.5 s.
#[cfg(feature = "cuda")]
const SHORT: Timing = Timing {
    dt: 0.25,
    decisions: &[0.25, 0.5, 0.75],
    gate_rate: 1.2,
    end: 0.75,
};

/// A pond joined to a sump by a submerged orifice, fed by a pipe and by the reach's right end.
///
/// An actuated gate drains the pond onto the surface. A controlled pump lifts the sump onto the
/// surface through a free outlet. The pond starts at its upper gate threshold, so the first
/// decision opens the gate. The gate travels 0.3 m at the actuator rate of `timing`.
fn case(timing: &Timing) -> pools_fixture::Result<Case> {
    let mut case = Case::tanks(0.0, 0.5);
    case.ponds = vec![
        (pond("pond", 0.0, 8.0, 2.0)?, 0.5 * 8.0),
        (pond("sump", 0.0, 4.0, 2.0)?, 0.45 * 4.0),
    ];
    case.inflows = vec![inflow_binding("feed", "pond")];
    case.pool_ends = vec![pool_end("outfall", ChannelEnd::Right, "pond")];
    case.structures = vec![
        structure(
            "orifice",
            orifice_law(0.0, 0.2, true)?,
            DirectionPolicy::Bidirectional,
            pools_fixture::manual(hydr::structures::StructureOperatingPoint::Passive),
            (pool("pond"), 0.0),
            (pool("sump"), 0.0),
        ),
        structure(
            "outlet",
            gate_law(0.0)?,
            DirectionPolicy::Bidirectional,
            SurfaceChannelOperationOwner::Actuated,
            (pool("pond"), 0.0),
            (surface(), 0.0),
        ),
        structure(
            "lift",
            pump_law(0.1, Some(1.0))?,
            DirectionPolicy::AtoBOnly,
            SurfaceChannelOperationOwner::ControlledPump,
            (pool("sump"), 0.0),
            (surface(), 0.0),
        ),
    ];
    let schedule = (0.0, timing.decisions, 20.0);
    case.operating = Some(SurfaceChannelOperating {
        gates: Some(controlled_gate(
            "outlet",
            [0.4, 0.5],
            ([0.0, 0.3], timing.gate_rate),
            schedule,
        )),
        pumps: Some(controlled_pump(
            "lift",
            [0.3, 0.4],
            [
                ControlCommand::PumpStoppedIsolated,
                ControlCommand::PumpRunning { speed_ratio: 1.0 },
            ],
            schedule,
        )),
        ..operating(vec![
            signal("outlet", pool("pond")),
            signal("lift", pool("sump")),
        ])
    });
    Ok(case)
}

/// Forcing from `start` to `end`: the pipe inflow, zero caller lateral rates, and a wall at the
/// reach end no pool end owns.
fn forcing(setup: &SurfaceChannelSetup, start: f64, end: f64) -> SurfaceChannelForcingInterval {
    let owned = |side: ChannelEnd| setup.pool_ends.iter().any(|e| e.end == side);
    let policy = |side| (!owned(side)).then_some(ChannelBoundary::Wall);
    let rate = |rate_m3_per_s| WaterAmount::Rate { rate_m3_per_s };
    SurfaceChannelForcingInterval {
        start_s: start,
        end_s: end,
        reaches: vec![SurfaceChannelReachForcing {
            reach_id: REACH.into(),
            left: policy(ChannelEnd::Left),
            right: policy(ChannelEnd::Right),
            lateral_inputs: vec![
                ChannelLateralInput::Inflow {
                    boundary_id: INFLOW.into(),
                    amount: rate(0.0),
                    longitudinal_velocity_m_per_s: 0.0,
                },
                ChannelLateralInput::Withdrawal {
                    boundary_id: WITHDRAWAL.into(),
                    amount: rate(0.0),
                },
            ],
        }],
        pool_sources: None,
        pool_inflows: vec![inflow_input("feed", rate(0.05))],
    }
}

/// Runs the case on one client and returns the state text after every accepted interval.
///
/// Fails unless the run crosses the relay decisions and the gate arrival, and every structure
/// and the pool end carry water.
macro_rules! pool_execution {
    ($runtime:ty, $client:expr, $timing:expr) => {{
        use hydr::coupling::SurfaceChannelBatch;

        let timing: &Timing = $timing;
        let (setup, initial) = case(timing)?.parts()?;
        let mut batch = SurfaceChannelBatch::<$runtime>::new($client, setup, initial)?;
        let mut states = Vec::new();
        while batch.time() < timing.end {
            let end = (batch.time() + timing.dt).min(timing.end);
            let held = forcing(batch.setup(), batch.time(), end);
            batch = batch.trial_to(end, &held)?.into_next();
            let pools = batch.pools().map(|pools| pools.states()).transpose()?;
            let reach = batch.reach(REACH).ok_or("missing reach")?.state()?;
            states.push(format!(
                "{} {} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
                batch.time(),
                batch.accepted_steps(),
                batch.snapshot_data()?,
                batch.ledger()?,
                batch.structure_ledgers()?,
                batch.pool_end_ledgers()?,
                batch.pool_inflow_ledgers()?,
                batch.link_ledgers()?,
                pools,
                batch.surface().state()?,
                reach,
                batch.last_step()?,
            ));
        }
        for ledger in batch.structure_ledgers()? {
            assert!(
                ledger.a_to_b_m3 > 0.0,
                "structure {} carried no water: {ledger:?}",
                ledger.structure_id
            );
        }
        let ends = batch.pool_end_ledgers()?;
        assert!(
            ends.iter()
                .any(|e| e.reach_to_pool_m3 > 0.0 || e.pool_to_reach_m3 > 0.0),
            "{ends:?}"
        );
        states
    }};
}

#[test]
fn selected_cpu_client_executes_surface_channel_pools_with_structures() -> Result {
    let states = pool_execution!(runtime::Runtime, runtime::client(), &LONG);
    assert!(states.len() >= 16);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_matches_cpu_surface_channel_pools_with_structures() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = pool_execution!(runtime::Runtime, runtime::client(), &SHORT);
    let cuda = pool_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?, &SHORT);
    assert_eq!(cpu.len(), cuda.len(), "accepted interval count");
    for (interval, (cpu, cuda)) in cpu.iter().zip(&cuda).enumerate() {
        super::same_bits(cpu, cuda)
            .map_err(|error| format!("interval {}: {error}", interval + 1))?;
    }
    Ok(())
}

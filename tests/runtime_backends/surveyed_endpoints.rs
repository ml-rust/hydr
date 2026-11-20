//! Surveyed endpoint attachments on configured runtime clients.
//!
//! Frozen criterion, "CUDA":
//! - Each run completes on the client.
//! - Each run restarts bit for bit: restore and import continue to the uninterrupted result.
//! - Cell volumes and momenta of the CUDA run equal the CPU run bit for bit, every interval.
//! - Ledger residuals stay within their bounds. The CPU and CUDA ledgers differ by at most the
//!   sum of the two bounds.
//!
//! Cases: a standalone `Stage` and `FluxPort`, a surface port on a surveyed rectangle with a
//! bank link, a surface pool end on three bands, and a channel/pool attachment on three bands.

/// Standalone surveyed reach fixtures.
#[allow(dead_code)]
#[path = "../channel_surveyed"]
mod standalone {
    #[path = "endpoints_support.rs"]
    pub mod endpoints_support;
    #[path = "support.rs"]
    pub mod support;
    #[path = "../support/surveyed_profiles.rs"]
    pub mod surveyed_profiles;
}

/// Surface/channel fixtures.
#[allow(dead_code, unused_imports)]
mod surface {
    pub(crate) use crate::surface_channel_pools::pools_fixture;
    pub(crate) use crate::surface_channels::{fixture, ports_fixture};
}

/// Channel/pool fixtures and the frozen three-band references.
#[allow(dead_code, unused_imports)]
#[path = "../channel_pools"]
mod pools {
    pub use super::standalone::surveyed_profiles;
    pub(crate) use crate::channel_pools::{fixture, reference};
    #[path = "surveyed_fixture.rs"]
    pub mod surveyed_fixture;
    #[path = "surveyed_reference.rs"]
    pub mod surveyed_reference;
}

use super::{Result, runtime};
use hydr::channel::{ChannelBoundary, ChannelConfig, ChannelFluxPort, ChannelSurveyedConfig};
use standalone::endpoints_support::Section;
use standalone::surveyed_profiles::THREE_BANDS;
use surface::ports_fixture::{ALPHA, BETA, Scenario, west};

/// One recorded value: its interval, field, cell, and bits. Only the CUDA comparison reads it.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Entry {
    interval: usize,
    field: &'static str,
    cell: usize,
    bits: u64,
}

/// What a run records: cell values per interval and one ledger per interval.
#[derive(Default)]
struct Run {
    entries: Vec<Entry>,
    ledgers: Vec<[f64; 3]>,
}

impl Run {
    fn cells(&mut self, field: &'static str, values: impl IntoIterator<Item = f64>) {
        let interval = self.ledgers.len();
        for (cell, value) in values.into_iter().enumerate() {
            let bits = value.to_bits();
            self.entries.push(Entry {
                interval,
                field,
                cell,
                bits,
            });
        }
    }

    /// Records storage, residual, and the admitted residual bound, then closes the interval.
    fn ledger(&mut self, storage: f64, residual: f64, bound: f64) {
        assert!(
            residual.abs() <= bound,
            "interval {}: residual {residual:e} exceeds bound {bound:e}",
            self.ledgers.len()
        );
        self.ledgers.push([storage, residual, bound]);
    }
}

/// Requires equal debug text, so equal bits, between a restarted and an uninterrupted result.
fn same_text<T: std::fmt::Debug>(left: &T, right: &T, what: &str) {
    assert_eq!(format!("{left:?}"), format!("{right:?}"), "{what}");
}

/// Requires equal cell bits and ledgers within the sum of both bounds.
#[cfg(feature = "cuda")]
fn same(name: &str, cpu: &Run, cuda: &Run) -> Result {
    if cpu.entries.len() != cuda.entries.len() || cpu.ledgers.len() != cuda.ledgers.len() {
        return Err(format!("{name}: CPU and CUDA record different counts").into());
    }
    for (a, b) in cpu.entries.iter().zip(&cuda.entries) {
        if a.bits != b.bits {
            return Err(format!(
                "{name}: interval {} field {} cell {}: CPU {:e} vs CUDA {:e}",
                a.interval,
                a.field,
                a.cell,
                f64::from_bits(a.bits),
                f64::from_bits(b.bits)
            )
            .into());
        }
    }
    for (interval, (a, b)) in cpu.ledgers.iter().zip(&cuda.ledgers).enumerate() {
        let allowance = a[2] + b[2];
        for (field, index) in [("storage", 0), ("residual", 1)] {
            if (a[index] - b[index]).abs() > allowance {
                return Err(format!(
                    "{name}: interval {interval} ledger {field}: CPU {:e} vs CUDA {:e}, allowance {allowance:e}",
                    a[index], b[index]
                )
                .into());
            }
        }
    }
    Ok(())
}

const CELLS: usize = 8;
const LENGTH: f64 = 2.5;
const DEPTH: f64 = 1.5;
const SPAN: f64 = 0.25;

/// Intervals before the checkpoint, and after it.
const HALF: usize = 1;

fn section() -> Section {
    Section::new(THREE_BANDS, 1.0)
}

/// A `Stage` below the rest stage at the left end, and a `FluxPort` entering water at the right.
fn ends(section: &Section) -> Result<[ChannelBoundary; 2]> {
    let port = ChannelFluxPort::new(
        -0.3,
        section.gravity * section.hydrostatic(DEPTH),
        4.0 * section.sound(DEPTH),
        Some(section.invariant(DEPTH)),
        "test provider of an entering surveyed flux",
    )?;
    Ok([
        ChannelBoundary::Stage { elevation_m: 0.81 },
        ChannelBoundary::FluxPort(port),
    ])
}

/// Three-band reach under `Stage` and `FluxPort`, checked every interval, then restarted.
macro_rules! standalone_run {
    ($runtime:ty, $client:expr) => {{
        use hydr::channel::ChannelReach;
        use standalone::support;

        let client = $client;
        let section = section();
        let ends = ends(&section)?;
        let origin = support::reach::<$runtime>(
            client.clone(),
            section.definition(),
            &[LENGTH; CELLS],
            &[0.0; CELLS],
            &[section.area(DEPTH); CELLS],
            &[0.0; CELLS],
            ChannelConfig {
                gravity: section.gravity,
                ..ChannelConfig::default()
            },
            ChannelSurveyedConfig::default(),
        )?;
        let advance = |mut current: ChannelReach<$runtime>,
                       count: usize,
                       run: &mut Run|
         -> Result<ChannelReach<$runtime>> {
            for _ in 0..count {
                let (start, target) = (current.time(), current.time() + SPAN);
                let held = support::forcing(start, target, ends.clone());
                current = current.step(target, &held)?.next;
                assert!(current.time() > start);
                support::check_water(&current)?;
                let state = current.state()?;
                run.cells("volume", state.iter().map(|c| c.volume_m3));
                run.cells(
                    "momentum",
                    state.iter().map(|c| c.longitudinal_momentum_m4_per_s),
                );
                let ledger = current.ledger();
                let scale = ledger
                    .initial_storage_m3
                    .max(ledger.storage_m3)
                    .max(ledger.boundary_in_m3)
                    .max(ledger.boundary_out_m3)
                    .max(1.0);
                let config = current.config();
                run.ledger(
                    ledger.storage_m3,
                    ledger.residual_m3,
                    config.volume_absolute + config.volume_relative * scale,
                );
            }
            Ok(current)
        };
        let mut run = Run::default();
        let before = origin.snapshot_data()?;
        let current = advance(origin.clone(), HALF, &mut run)?;
        let saved = current.snapshot_data()?;
        let whole = advance(current.clone(), HALF, &mut run)?.snapshot_data()?;
        let trusted = origin.restore(&current.snapshot())?;
        let cold = origin.import_snapshot(&saved)?;
        same_text(&trusted.snapshot_data()?, &saved, "restored checkpoint");
        same_text(&cold.snapshot_data()?, &saved, "imported checkpoint");
        for restarted in [trusted, cold] {
            let resumed = advance(restarted, HALF, &mut Run::default())?.snapshot_data()?;
            same_text(&resumed, &whole, "continued restart");
        }
        same_text(&origin.snapshot_data()?, &before, "origin");
        run
    }};
}

#[test]
fn selected_cpu_client_executes_surveyed_stage_and_flux_port_with_restart() -> Result {
    let run = standalone_run!(runtime::Runtime, runtime::client());
    assert_eq!(run.ledgers.len(), 2 * HALF);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_matches_cpu_surveyed_stage_and_flux_port() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = standalone_run!(runtime::Runtime, runtime::client());
    let cuda = standalone_run!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    same("standalone", &cpu, &cuda)
}

/// Requested duration of every port interval, in s.
const PORT_DT: f64 = 0.25;

/// A surveyed reach above its floodplain drains through the west port and the bank link.
fn port_case() -> Scenario {
    let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]).with_surveyed(true);
    case.alpha = surface::ports_fixture::Reach::rest(0.5, 1.75);
    case
}

/// Surface port on a surveyed rectangle with a bank link, checked every interval, then restarted.
macro_rules! port_run {
    ($runtime:ty, $client:expr) => {{
        use hydr::coupling::SurfaceChannelBatch;

        let client = $client;
        let (setup, initial) = port_case().parts()?;
        let build =
            || SurfaceChannelBatch::<$runtime>::new(client.clone(), setup.clone(), initial.clone());
        let advance = |mut batch: SurfaceChannelBatch<$runtime>,
                       count: usize,
                       run: &mut Run|
         -> Result<SurfaceChannelBatch<$runtime>> {
            for _ in 0..count {
                let (start, end) = (batch.time(), batch.time() + PORT_DT);
                let held = surface::ports_fixture::forcing(batch.setup(), start, end);
                batch = batch.trial_to(end, &held)?.into_next();
                let step = batch.last_step()?.ok_or("missing interval receipt")?;
                let port = step.ports.first().ok_or("missing port receipt")?;
                assert!(port.paired_gap_m3.abs() <= port.paired_bound_m3, "{port:?}");
                let surface = batch.surface().state()?;
                run.cells("surface depth", surface.cells().iter().map(|c| c.h));
                run.cells("surface qx", surface.cells().iter().map(|c| c.qx));
                for id in [ALPHA, BETA] {
                    let cells = batch.reach(id).ok_or("missing reach")?.state()?;
                    run.cells("volume", cells.iter().map(|c| c.volume_m3));
                    run.cells(
                        "momentum",
                        cells.iter().map(|c| c.longitudinal_momentum_m4_per_s),
                    );
                }
                let ledger = batch.ledger()?;
                run.ledger(ledger.storage_m3, ledger.residual_m3, ledger.bound_m3);
            }
            Ok(batch)
        };
        let mut run = Run::default();
        let middle = advance(build()?, HALF, &mut run)?;
        let trusted = middle.snapshot();
        let data = middle.snapshot_data()?;
        let whole = advance(middle, HALF, &mut run)?;
        let expected = whole.snapshot_data()?;
        let ledger = &whole.port_ledgers()?[0];
        assert!(ledger.surface_to_channel_m3 > 0.0 || ledger.channel_to_surface_m3 > 0.0);
        let restored = build()?.restore(&trusted)?;
        let imported = build()?.import_snapshot(&data)?;
        for (name, batch) in [("restored", restored), ("imported", imported)] {
            same_text(&batch.snapshot_data()?, &data, name);
            let continued = advance(batch, HALF, &mut Run::default())?;
            same_text(&continued.snapshot_data()?, &expected, name);
        }
        run
    }};
}

#[test]
fn selected_cpu_client_executes_surveyed_port_with_bank_link_and_restart() -> Result {
    let run = port_run!(runtime::Runtime, runtime::client());
    assert_eq!(run.ledgers.len(), 2 * HALF);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_matches_cpu_surveyed_port_with_bank_link() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = port_run!(runtime::Runtime, runtime::client());
    let cuda = port_run!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    same("surface port", &cpu, &cuda)
}

#[path = "surveyed_endpoint_pools.rs"]
mod pool_cases;

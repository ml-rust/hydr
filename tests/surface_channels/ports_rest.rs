//! Lake at rest through reach-end ports: flat and stepped beds, one face and four faces,
//! water below the higher bed, and mirror symmetry of a four-face port.

use crate::fixture::Result;
use crate::ports_fixture::{
    ALPHA, BETA, Batch, LENGTH, Reach, Scenario, balanced, last, port, reach_cells, step, x_face,
    y_face,
};
use hydr::channel::ChannelEnd;
use hydr::coupling::{SurfaceChannelPort, SurfaceChannelPortRegime};

/// Requested duration of every interval, in s. The CFL limit shortens it.
const DT: f64 = 0.1;

/// Accepted intervals of every run.
const INTERVALS: usize = 50;

/// Frozen bound on every `|q|`, `|Q|`, and stage change at rest.
const REST: f64 = 1e-12;

/// Frozen relative bound between mirror cells.
const MIRROR: f64 = 1e-12;

/// Surface bed of every case, in m.
const SURFACE_BED: f64 = 1.0;

/// A port of `alpha`'s left end and the `alpha` cell size that matches its face width.
struct Shape {
    name: &'static str,
    port: SurfaceChannelPort,
    alpha_size: [f64; 2],
}

/// One 2 m west face, and the four 2 m north faces.
fn shapes() -> [Shape; 2] {
    [
        Shape {
            name: "one face",
            port: port("p", ALPHA, ChannelEnd::Left, vec![x_face(0, 0)]),
            alpha_size: [2.0, LENGTH],
        },
        Shape {
            name: "four faces",
            port: port(
                "p",
                ALPHA,
                ChannelEnd::Left,
                (0..4).map(|column| y_face(column, 2)).collect(),
            ),
            alpha_size: [8.0, LENGTH],
        },
    ]
}

/// `case` with the port and `alpha` cell size of `shape`.
fn shaped(mut case: Scenario, shape: &Shape) -> Scenario {
    case.ports = vec![shape.port.clone()];
    case.alpha_size = shape.alpha_size;
    case
}

/// Fails unless every wet surface cell and reach cell of `batch` rests at `stage`.
fn at_rest(batch: &Batch, stage: f64, name: &str) -> Result {
    for cell in batch.surface().state()?.cells() {
        assert!(
            cell.qx.abs() <= REST && cell.qy.abs() <= REST,
            "{name}: {cell:?}"
        );
        assert!(
            (cell.h + SURFACE_BED - stage).abs() <= REST,
            "{name}: {cell:?}"
        );
    }
    for id in [ALPHA, BETA] {
        for cell in reach_cells(batch, id)? {
            assert!(
                cell.discharge_m3_per_s.abs() <= REST,
                "{name} {id}: {cell:?}"
            );
            let wet = cell.stage_m.ok_or("every reach cell is wet")?;
            assert!((wet - stage).abs() <= REST, "{name} {id}: {cell:?}");
        }
    }
    Ok(())
}

#[test]
fn a_lake_at_rest_stays_at_rest_on_flat_and_stepped_beds_through_one_and_four_faces() -> Result {
    let stage = 1.6;
    // The reach bed equals the surface bed, lies below it, and lies above it. The bank crest
    // at 1.2 m stays above every bed and below the stage.
    for alpha_bed in [SURFACE_BED, 0.5, 1.125] {
        for shape in shapes() {
            let name = format!("{} on bed {alpha_bed}", shape.name);
            let case = shaped(
                Scenario::rest(SURFACE_BED, stage, alpha_bed, Vec::new()),
                &shape,
            );
            let mut batch = case.build()?;
            for _ in 0..INTERVALS {
                batch = step(&batch, DT)?;
                let transfer = &last(&batch)?.ports[0];
                assert_eq!(
                    transfer.regime,
                    SurfaceChannelPortRegime::Subcritical,
                    "{name}"
                );
                assert!(transfer.paired_gap_m3.abs() <= transfer.paired_bound_m3);
            }
            assert_eq!(batch.accepted_steps(), INTERVALS, "{name}");
            at_rest(&batch, stage, &name)?;
            balanced(&batch)?;
        }
    }
    Ok(())
}

#[test]
fn water_below_the_higher_bed_moves_nothing() -> Result {
    // The reach stage lies below the dry surface bed, then the surface stage lies below the
    // dry reach bed.
    let cases = [(0.8, 0.5), (1.3, 1.5)];
    for (stage, alpha_bed) in cases {
        for shape in shapes() {
            let name = format!("{} at stage {stage}", shape.name);
            let mut case = shaped(
                Scenario::rest(SURFACE_BED, stage, alpha_bed, Vec::new()),
                &shape,
            );
            case.link = false;
            let start = case.build()?;
            let surface = start.surface().state()?;
            let alpha = reach_cells(&start, ALPHA)?;
            let mut batch = start;
            for _ in 0..INTERVALS {
                batch = step(&batch, DT)?;
                let transfer = &last(&batch)?.ports[0];
                assert_eq!(transfer.regime, SurfaceChannelPortRegime::Dry, "{name}");
                assert_eq!(
                    transfer.signed_volume_m3.to_bits(),
                    0.0f64.to_bits(),
                    "{name}"
                );
            }
            let ledger = &batch.port_ledgers()?[0];
            assert_eq!(ledger.surface_to_channel_m3.to_bits(), 0.0f64.to_bits());
            assert_eq!(ledger.channel_to_surface_m3.to_bits(), 0.0f64.to_bits());
            let exchange = batch.surface().ledger();
            assert_eq!(exchange.exchange_in_m3.to_bits(), 0.0f64.to_bits());
            assert_eq!(exchange.exchange_out_m3.to_bits(), 0.0f64.to_bits());
            assert_eq!(batch.surface().state()?, surface, "{name}");
            assert_eq!(reach_cells(&batch, ALPHA)?, alpha, "{name}");
        }
    }
    Ok(())
}

/// `|a - b| / max(|a|, |b|)`, zero when both are zero.
fn relative(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs());
    if scale == 0.0 {
        0.0
    } else {
        (a - b).abs() / scale
    }
}

#[test]
fn a_four_face_port_keeps_mirror_cells_equal() -> Result {
    // The reach stands above the surface and drains across the whole north edge. The grid
    // mirrors about x = 4 m: column c of each row mirrors column 3 - c.
    let [_, four] = shapes();
    let mut case = shaped(Scenario::rest(SURFACE_BED, 1.6, 0.5, Vec::new()), &four);
    case.link = false;
    case.alpha = Reach::rest(0.5, 1.8);
    let mut batch = case.build()?;
    let mut moved = false;
    for _ in 0..INTERVALS {
        batch = step(&batch, DT)?;
        let cells = batch.surface().state()?.cells().to_vec();
        for row in 0..2 {
            for column in 0..2 {
                let a = cells[row * 4 + column];
                let b = cells[row * 4 + 3 - column];
                assert!(relative(a.h, b.h) <= MIRROR, "{a:?} against {b:?}");
                assert!(relative(a.qx, -b.qx) <= MIRROR, "{a:?} against {b:?}");
                assert!(relative(a.qy, b.qy) <= MIRROR, "{a:?} against {b:?}");
                moved |= a.qy != 0.0;
            }
        }
        balanced(&batch)?;
    }
    assert!(moved, "the port must move the surface water");
    Ok(())
}

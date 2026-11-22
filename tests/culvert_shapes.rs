//! Tangent-arc barrels: frozen independent references, construction, geometry, critical depth
//! and normal depth.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "culvert_shapes/reference_rows.rs"]
mod reference_rows;

#[path = "culvert_shapes/depth_rows.rs"]
mod depth_rows;

#[path = "culvert_shapes/geometry.rs"]
mod geometry;

#[path = "culvert_shapes/sweeps.rs"]
mod sweeps;

#[path = "culvert_shapes/layouts.rs"]
mod layouts;

#[path = "culvert_shapes/depths.rs"]
mod depths;

#[path = "culvert_shapes/depth_laws.rs"]
mod depth_laws;

use hydr::structures::{
    CriticalDepthStatus, CulvertBarrel, CulvertBarrelBatch, CulvertBarrelShape, NormalDepthStatus,
};
use reference_rows::{BARRELS, Barrel, GEOMETRY_SHA256, ROWS, Shape, State};
use runtime::{Runtime, client};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn inputs(entry: &Barrel) -> Vec<f64> {
    entry
        .inputs
        .iter()
        .map(|bits| f64::from_bits(*bits))
        .collect()
}

fn barrel(entry: &Barrel) -> Result<CulvertBarrel, Box<dyn std::error::Error>> {
    Ok(match (entry.shape, inputs(entry).as_slice()) {
        (Shape::HorizontalEllipse, &[b, d, r]) => CulvertBarrel::horizontal_ellipse(b, d, r)?,
        (Shape::VerticalEllipse, &[b, d, r]) => CulvertBarrel::vertical_ellipse(b, d, r)?,
        (Shape::PipeArch, &[b, d, c, r]) => CulvertBarrel::pipe_arch(b, d, c, r)?,
        (Shape::Arch, &[b, d]) => CulvertBarrel::arch(b, d)?,
        _ => return Err(format!("fixture barrel {}: wrong input count", entry.name).into()),
    })
}

#[test]
fn geometry_fixture_is_frozen_and_self_consistent() -> TestResult {
    assert_eq!(
        GEOMETRY_SHA256,
        "51debc4680d88121a9c5022973cb327111741b275140edeb215c00364661c7fb"
    );
    let mut previous: Option<(usize, f64)> = None;
    for row in ROWS {
        let entry = BARRELS.get(row.barrel).ok_or("row names no barrel")?;
        let rise = inputs(entry).get(1).copied().ok_or("barrel has no rise")?;
        let depth = f64::from_bits(row.depth);
        let expected = if depth == 0.0 {
            State::Dry
        } else if depth >= rise {
            State::Full
        } else {
            State::Free
        };
        assert_eq!(row.state, expected, "{} {}", entry.name, row.fraction);
        if let Some((barrel, last)) = previous
            && barrel == row.barrel
        {
            assert!(
                depth > last,
                "{} rows out of order at {}",
                entry.name,
                row.fraction
            );
        }
        previous = Some((row.barrel, depth));
        for ([lower, upper], width) in row.values.iter().zip(row.relative_width) {
            let (lower, upper) = (f64::from_bits(*lower), f64::from_bits(*upper));
            assert!(
                lower <= upper && lower >= 0.0,
                "{} {}",
                entry.name,
                row.fraction
            );
            assert!(
                width.is_finite() && width >= 0.0,
                "{} {}",
                entry.name,
                row.fraction
            );
            if row.state == State::Dry {
                assert_eq!(width, 0.0);
            }
        }
    }
    // Full rows of one barrel share area, perimeter and hydraulic radius.
    for (index, entry) in BARRELS.iter().enumerate() {
        let full: Vec<_> = ROWS
            .iter()
            .filter(|row| row.barrel == index && row.state == State::Full)
            .collect();
        assert_eq!(full.len(), 2, "{}", entry.name);
        for pair in full.windows(2) {
            let [a, b] = pair else { continue };
            assert_eq!(a.values[..4], b.values[..4], "{}", entry.name);
            assert!(b.values[4][0] > a.values[4][1], "{}", entry.name);
        }
    }
    Ok(())
}

#[test]
fn evidence_barrels_construct_and_report_their_inputs() -> TestResult {
    for entry in BARRELS {
        let built = barrel(entry)?;
        let values = inputs(entry);
        let shape = match entry.shape {
            Shape::HorizontalEllipse => CulvertBarrelShape::HorizontalEllipse,
            Shape::VerticalEllipse => CulvertBarrelShape::VerticalEllipse,
            Shape::PipeArch => CulvertBarrelShape::PipeArch,
            Shape::Arch => CulvertBarrelShape::Arch,
        };
        assert_eq!(built.shape(), shape, "{}", entry.name);
        assert_eq!(
            &values[..2],
            &[built.span_m(), built.rise_m()],
            "{}",
            entry.name
        );
        let given = built.given_radius_m().unwrap_or(&[]);
        assert_eq!(&values[2..], given, "{}", entry.name);
    }
    Ok(())
}

#[test]
fn depth_queries_answer_arc_rows_with_typed_outcomes() -> TestResult {
    let device = numr::runtime::RuntimeClient::device(&client()).clone();
    let column =
        |values: &[f64]| numr::tensor::Tensor::<Runtime>::from_slice(values, &[2], &device);
    for entry in BARRELS {
        let barrels = vec![CulvertBarrel::circular(1.0)?, barrel(entry)?];
        let batch = CulvertBarrelBatch::<Runtime>::new(client(), barrels)?;
        let rows = batch
            .critical_depth(&column(&[0.5, 0.0])?, 9.81)?
            .project()?;
        assert_eq!(rows[1].status, CriticalDepthStatus::Dry, "{}", entry.name);
        let flows = column(&[0.5, 0.5])?;
        let roughness = column(&[0.013, 0.013])?;
        let rows = batch
            .normal_depth(&flows, &roughness, &column(&[0.01, 0.0])?)?
            .project()?;
        assert_eq!(
            rows[1].status,
            NormalDepthStatus::NonPositiveSlope,
            "{}",
            entry.name
        );
        assert_eq!(rows[1].depth_m, None, "{}", entry.name);
    }
    Ok(())
}

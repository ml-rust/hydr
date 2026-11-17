use super::super::ConditionalBandSegments;
use super::{
    fixtures::{self, GapAuthority, GapRef, PanelKind, Path, RangePolicy, RangeRef, Sample},
    native::Snapshot,
};
use crate::channel::flow::surveyed::error::HydraulicResult;
use numr::{dtype::DType, runtime::Runtime};

fn range(actual: [f64; 2], reference: RangeRef, label: &str) {
    let minimum: fixtures::Bounds = reference.minimum;
    let minimum = minimum.map(f64::from_bits);
    let maximum = reference.maximum.map(f64::from_bits);
    assert!(
        actual[0].is_finite() && actual[1].is_finite() && actual[0] <= actual[1],
        "{label}: {actual:?}"
    );
    assert!(
        actual[0] <= minimum[0] && maximum[1] <= actual[1],
        "{label}: {actual:?} excludes minimum={minimum:?}, maximum={maximum:?}"
    );
    match reference.policy {
        RangePolicy::ExactZero => assert_eq!(actual, [0.0; 2], "{label}"),
        RangePolicy::ConditionedLocal | RangePolicy::OriginalGeometry => {
            let excess = [minimum[1] - actual[0], actual[1] - maximum[0]];
            for (endpoint, excess) in excess.into_iter().enumerate() {
                let allowance = f64::from_bits(reference.native_allowances[endpoint]);
                assert!(
                    excess <= allowance,
                    "{label}: endpoint={endpoint}, excess={}, allowance={allowance}, actual={actual:?}",
                    excess
                );
            }
        }
    }
    if reference.positive && minimum[0] > 0.0 {
        assert!(actual[0] > 0.0, "{label}: positive physical range");
    }
}

pub(super) fn sample(actual: &Snapshot, row: usize, reference: &Sample, label: &str) {
    assert_eq!(actual.fields.len(), fixtures::FIELD_ORDER.len());
    for (field, (values, reference)) in actual.fields.iter().zip(&reference.fields).enumerate() {
        range(
            values[row],
            *reference,
            &format!("{label}, field={}", fixtures::FIELD_ORDER[field]),
        );
    }
}

fn gap(actual: [f64; 2], reference: &GapRef, label: &str) {
    let expected = reference.bounds.map(f64::from_bits);
    assert!(
        actual[0].is_finite() && actual[1].is_finite() && actual[0] <= actual[1],
        "{label}: gap={actual:?}"
    );
    assert!(
        actual[0] <= expected[0] && expected[1] <= actual[1],
        "{label}: gap={actual:?} excludes {expected:?}"
    );
    let error = (actual[0] - expected[1])
        .abs()
        .max((actual[1] - expected[0]).abs());
    assert!(
        error <= f64::from_bits(reference.native_allowance),
        "{label}: gap error={error}, allowance={}",
        f64::from_bits(reference.native_allowance)
    );
    if reference.exact_zero {
        assert_eq!(actual, [0.0; 2], "{label}");
    }
    if reference.authority == GapAuthority::ConditionedLocal {
        if expected[0] > 0.0 {
            assert!(actual[0] > 0.0, "{label}: positive physical gap");
        }
        if expected[1] < 0.0 {
            assert!(actual[1] < 0.0, "{label}: negative physical gap");
        }
    }
}

pub(super) fn gaps<R: Runtime<DType = DType>>(
    segments: &ConditionalBandSegments<R>,
    raw_projection: &[[f64; 2]],
    path: &Path,
) -> HydraulicResult<()> {
    let low = segments.physical_gap().lower().try_to_vec::<f64>()?;
    let high = segments.physical_gap().upper().try_to_vec::<f64>()?;
    let raw_high = segments.retained_gap().high().try_to_vec::<f64>()?;
    let raw_low = segments.retained_gap().low().try_to_vec::<f64>()?;
    let radius = segments.retained_gap().abs_error().try_to_vec::<f64>()?;
    let zero = segments.zero_gap_mask().try_to_vec::<f64>()?;
    let active = segments.active_mask().try_to_vec::<f64>()?;
    let originals = segments.original_rows().try_to_vec::<i64>()?;
    let indices = segments.band_indices().try_to_vec::<i64>()?;
    let b = fixtures::MAXIMUM_BANDS;
    assert_eq!(segments.owners().numel(), 3 * b);
    for (repeated, original) in [2_i64, 0, 2].into_iter().enumerate() {
        for panel in &path.panels {
            let lane = repeated * b + panel.band_index;
            let label = format!(
                "{}, band={}, selected_row={repeated}",
                path.id, panel.band_index
            );
            assert_eq!(originals[lane], original, "{label}");
            assert_eq!(indices[lane], panel.band_index as i64, "{label}");
            assert_eq!(panel.ordinary_active, panel.kind == PanelKind::OrdinaryWet);
            gap([low[lane], high[lane]], &panel.gap, &label);
            gap(
                raw_projection[lane],
                &panel.gap,
                &format!("{label}, retained"),
            );
            if panel.kind == PanelKind::Zero {
                assert!(
                    panel.shared_boundary_identity || panel.original_source_equals_target,
                    "{label}"
                );
                if panel.shared_boundary_identity {
                    assert_eq!(panel.source_clip, panel.target_clip, "{label}");
                }
                assert_eq!(
                    [raw_high[lane], raw_low[lane], radius[lane], active[lane]],
                    [0.0; 4],
                    "{label}"
                );
                assert_eq!(zero[lane], 1.0, "{label}");
                assert!(
                    panel.points.is_empty()
                        && panel.intervals.is_empty()
                        && panel.endpoint_geometry.is_empty()
                );
            }
        }
        for &band in &path.padded_bands {
            let lane = repeated * b + band;
            assert_eq!(
                [
                    low[lane],
                    high[lane],
                    raw_high[lane],
                    raw_low[lane],
                    radius[lane],
                    active[lane]
                ],
                [0.0; 6],
                "{}, padded band={band}",
                path.id
            );
            assert_eq!(zero[lane], 1.0);
        }
    }
    Ok(())
}

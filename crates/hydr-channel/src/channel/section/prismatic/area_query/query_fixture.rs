use super::fixtures::{AnchorKind, Chart};
use super::test_support::Result;
use super::{AreaAnchor, AreaDirection, SupportedAreaQuery};
use numr::{
    dtype::DType,
    runtime::{Device, Runtime},
    tensor::Tensor,
};

pub(super) fn float_fields<R: Runtime<DType = DType>>(
    value: &SupportedAreaQuery<R>,
) -> Vec<&Tensor<R>> {
    let q = &value.query;
    let mut fields = vec![
        q.scale.high(),
        q.scale.low(),
        q.scale.abs_error(),
        q.parameter.lower(),
        q.parameter.upper(),
        q.offsets.lower.high(),
        q.offsets.lower.low(),
        q.offsets.lower.abs_error(),
        q.offsets.upper.high(),
        q.offsets.upper.low(),
        q.offsets.upper.abs_error(),
        &value.candidate,
    ];
    for interval in [
        &value.area,
        &value.above_start,
        &value.below_end,
        &value.lower_endpoint.above_start,
        &value.lower_endpoint.below_end,
        &value.upper_endpoint.above_start,
        &value.upper_endpoint.below_end,
    ] {
        fields.extend([interval.lower(), interval.upper()]);
    }
    if let AreaAnchor::Source { area, coordinates } = &q.anchor {
        fields.extend([
            area.high(),
            area.low(),
            area.abs_error(),
            coordinates.candidate_mask(),
        ]);
        for interval in [
            coordinates.area_above_start_m2(),
            coordinates.area_below_end_m2(),
            coordinates.depth_above_start_m(),
            coordinates.depth_below_end_m(),
        ] {
            fields.extend([interval.lower(), interval.upper()]);
        }
    }
    fields
}

pub(super) fn integer_fields<R: Runtime<DType = DType>>(
    value: &SupportedAreaQuery<R>,
) -> Vec<&Tensor<R>> {
    let mut fields = vec![&value.query.owners, &value.query.original_rows];
    if let AreaAnchor::BandStart(index) | AreaAnchor::BandEnd(index) = &value.query.anchor {
        fields.push(index);
    }
    fields
}

#[derive(Debug, PartialEq)]
pub(super) struct Snapshot {
    floats: Vec<(Vec<usize>, Vec<u64>)>,
    integers: Vec<(Vec<usize>, Vec<i64>)>,
    anchor: &'static str,
    add: bool,
    bands: usize,
}

pub(super) fn snapshot<R: Runtime<DType = DType>>(
    value: &SupportedAreaQuery<R>,
) -> Result<Snapshot> {
    let floats = float_fields(value)
        .into_iter()
        .map(|v| {
            Ok((
                v.shape().to_vec(),
                v.try_to_vec::<f64>()?
                    .into_iter()
                    .map(f64::to_bits)
                    .collect(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let integers = integer_fields(value)
        .into_iter()
        .map(|v| Ok((v.shape().to_vec(), v.try_to_vec::<i64>()?)))
        .collect::<Result<Vec<_>>>()?;
    let anchor = match &value.query.anchor {
        AreaAnchor::Source { .. } => "source",
        AreaAnchor::Zero => "zero",
        AreaAnchor::BandStart(_) => "band start",
        AreaAnchor::BandEnd(_) => "band end",
        AreaAnchor::Cap => "cap",
    };
    Ok(Snapshot {
        floats,
        integers,
        anchor,
        add: matches!(value.query.direction, AreaDirection::Add),
        bands: value.query.band_count,
    })
}

pub(super) fn shape<R: Runtime<DType = DType>>(
    proved: &Tensor<R>,
    owner: &Tensor<R>,
    count: usize,
) {
    assert_eq!(proved.shape(), &[count]);
    assert_eq!(proved.dtype(), DType::F64);
    assert_eq!(owner.ndim(), 1);
    assert_eq!(owner.dtype(), DType::I64);
    assert!(proved.device().is_same(owner.device()));
}

pub(super) fn rectangle_chart(kind: AnchorKind, add: bool, lower: f64, upper: f64) -> Chart {
    Chart {
        owner: 1,
        kind,
        band: 0,
        source: [2.0_f64.to_bits(), 0, 0],
        add,
        scale: [1.0_f64.to_bits(), 0, 0],
        parameter: [lower.to_bits(), upper.to_bits()],
    }
}

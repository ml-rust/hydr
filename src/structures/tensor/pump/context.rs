//! Grouped immutable pump reference curves and exact per-element orientation/bounds.
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::{
    DirectionPolicy, StructureElement, StructureError, StructureInputErrorKind, StructureLaw,
};
use numr::tensor::Tensor;

use std::collections::BTreeMap;

#[derive(Debug)]
pub(crate) struct TensorPumpGroup<R: Runtime<DType = DType>> {
    pub slots: Vec<usize>,
    pub indices: Tensor<R>,
    pub heads: Tensor<R>,
    pub flows: Tensor<R>,
    pub minimum: Tensor<R>,
    pub maximum: Tensor<R>,
    pub zero_start: Tensor<R>,
    pub has_zero: Tensor<R>,
    pub zero: Tensor<R>,
}
#[derive(Debug)]
pub(crate) struct TensorPumpContext<R: Runtime<DType = DType>> {
    pub mask: Tensor<R>,
    pub orientation: Tensor<R>,
    pub minimum_suction: Tensor<R>,
    pub minimum_speed: Tensor<R>,
    pub maximum_speed: Tensor<R>,
    pub groups: Vec<TensorPumpGroup<R>>,
    pub all_indices: Tensor<R>,
}
impl<R: Runtime<DType = DType>> TensorPumpContext<R> {
    pub(crate) fn new(
        elements: &[StructureElement],
        device: &R::Device,
    ) -> Result<Option<Self>, StructureError> {
        if !elements
            .iter()
            .any(|e| matches!(e.law, StructureLaw::Pump(_)))
        {
            return Ok(None);
        }
        let n = elements.len();
        let mut groups = BTreeMap::<_, Vec<usize>>::new();
        let (mut mask, mut orientation, mut suction, mut speed_min, mut speed_max) = (
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![1.0; n],
            vec![1.0; n],
        );
        for (index, e) in elements.iter().enumerate() {
            if let StructureLaw::Pump(p) = &e.law {
                if e.direction == DirectionPolicy::Bidirectional {
                    return Err(StructureError::input(
                        StructureInputErrorKind::Definition,
                        "pump requires a fixed A-to-B or B-to-A orientation",
                    )
                    .at(index));
                }
                mask[index] = 1.0;
                orientation[index] = if e.direction == DirectionPolicy::AtoBOnly {
                    1.0
                } else {
                    -1.0
                };
                suction[index] = p.minimum_suction_stage_m();
                let (lo, hi) = p.speed_policy().range();
                speed_min[index] = lo;
                speed_max[index] = hi;
                groups.entry(p.curve().key()).or_default().push(index);
            }
        }
        let tensor = |v: &[f64]| Tensor::<R>::from_slice(v, &[v.len()], device);
        let indices = |v: &[usize]| -> Result<Tensor<R>, StructureError> {
            let values = v
                .iter()
                .map(|i| {
                    i64::try_from(*i).map_err(|_| {
                        StructureError::input(
                            StructureInputErrorKind::Definition,
                            "pump count exceeds I64 indexing",
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Tensor::<R>::from_slice(&values, &[values.len()], device)?)
        };
        let mut cached = Vec::with_capacity(groups.len());
        let mut all = Vec::new();
        for ((_, knots), slots) in groups {
            let heads: Vec<_> = knots.iter().map(|(h, _)| f64::from_bits(*h)).collect();
            let flows: Vec<_> = knots.iter().map(|(_, q)| f64::from_bits(*q)).collect();
            let first = *heads.first().ok_or_else(|| {
                StructureError::input(
                    StructureInputErrorKind::Definition,
                    "empty checked pump curve",
                )
            })?;
            let last = *heads.last().ok_or_else(|| {
                StructureError::input(
                    StructureInputErrorKind::Definition,
                    "empty checked pump curve",
                )
            })?;
            let zero = flows.iter().position(|q| *q == 0.0);
            let zero_start = zero.map_or(last, |i| heads[i]);
            let count = slots.len();
            all.extend_from_slice(&slots);
            cached.push(TensorPumpGroup {
                indices: indices(&slots)?,
                heads: tensor(&heads)?,
                flows: tensor(&flows)?,
                minimum: tensor(&vec![first; count])?,
                maximum: tensor(&vec![last; count])?,
                zero_start: tensor(&vec![zero_start; count])?,
                has_zero: tensor(&vec![if zero.is_some() { 1.0 } else { 0.0 }; count])?,
                zero: tensor(&vec![0.0; count])?,
                slots,
            });
        }
        Ok(Some(Self {
            mask: tensor(&mask)?,
            orientation: tensor(&orientation)?,
            minimum_suction: tensor(&suction)?,
            minimum_speed: tensor(&speed_min)?,
            maximum_speed: tensor(&speed_max)?,
            groups: cached,
            all_indices: indices(&all)?,
        }))
    }
}

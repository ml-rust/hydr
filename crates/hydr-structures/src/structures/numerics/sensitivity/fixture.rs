//! Centred-difference checks of the head sensitivity over sampled heads.
use crate::structures::coupling::CouplingStructures;
use crate::structures::{
    DirectionPolicy, StructureElement, StructureLaw, StructureOperatingPoint, StructureOperation,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

pub(crate) type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// Centred-difference step as a fraction of the lane's nearest regime distance.
pub(crate) const STEP: f64 = 1e-4;
/// Smallest distance a sampled lane keeps from every regime boundary, in m.
pub(crate) const MARGIN: f64 = 0.01;
/// Frozen before the first run: every centred difference stays within `s |Q| (1 + SLACK)`.
///
/// The step keeps truncation near `STEP²` of the slope. Rounding of `Q` adds a few `ε |Q|`
/// over a step of at least `STEP × MARGIN`. Both lie far below `SLACK`.
const SLACK: f64 = 1e-6;
/// Bed of every endpoint, below every crest, sill, and suction stage in these cases.
pub(crate) const BED: f64 = -1.0;

/// Endpoint stages, wet flags, and the centred-difference step of one lane.
#[derive(Clone, Copy)]
pub(crate) struct Lane {
    pub stage: [f64; 2],
    pub wet: [bool; 2],
    pub step: f64,
}

/// Lanes over the grid `values²` whose wet heads keep `MARGIN` from each `boundary` and from
/// each other.
pub(crate) fn lanes(values: &[f64], boundaries: &[f64], wet: [bool; 2]) -> Vec<Lane> {
    let nearest = |stage: f64, from: f64| {
        boundaries
            .iter()
            .map(|z| (stage - z).abs())
            .fold(from, f64::min)
    };
    let mut lanes = Vec::new();
    for &a in values {
        for &b in values {
            let mut distance = nearest(a, f64::MAX);
            if wet[1] {
                distance = nearest(b, distance).min((a - b).abs());
            }
            if distance >= MARGIN {
                lanes.push(Lane {
                    stage: [a, b],
                    wet,
                    step: STEP * distance,
                });
            }
        }
    }
    lanes
}

/// `count` points spread over `(low, high)`, off any round value.
pub(crate) fn grid(low: f64, high: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| low + (high - low) * (i as f64 + 0.37) / count as f64)
        .collect()
}

/// One copy of `law(index)` per lane, each with its own direction and operation.
pub(crate) fn structures(
    law: impl Fn(usize) -> (StructureLaw, DirectionPolicy, StructureOperatingPoint),
    count: usize,
) -> TestResult<CouplingStructures<CpuRuntime>> {
    let (client, elements, operations) = parts(law, count);
    Ok(CouplingStructures::new(client, elements, operations)?)
}

/// [`structures`] through the constructor that admits culverts.
pub(crate) fn culvert_structures(
    law: impl Fn(usize) -> (StructureLaw, DirectionPolicy, StructureOperatingPoint),
    count: usize,
) -> TestResult<CouplingStructures<CpuRuntime>> {
    let (client, elements, operations) = parts(law, count);
    Ok(CouplingStructures::with_culverts(
        client, elements, operations,
    )?)
}

/// Client, elements and operations of [`structures`].
type Parts = (CpuClient, Vec<StructureElement>, Vec<StructureOperation>);

fn parts(
    law: impl Fn(usize) -> (StructureLaw, DirectionPolicy, StructureOperatingPoint),
    count: usize,
) -> Parts {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let (mut elements, mut operations) = (Vec::new(), Vec::new());
    for index in 0..count {
        let (law, direction, operating_point) = law(index);
        elements.push(StructureElement {
            id: format!("e{index}"),
            endpoint_a: "a".into(),
            endpoint_b: "b".into(),
            datum: "analytical".into(),
            direction,
            law,
            condition: crate::structures::StructureCondition::AsBuilt,
        });
        operations.push(StructureOperation {
            structure_id: format!("e{index}"),
            operating_point,
        });
    }
    (client, elements, operations)
}

/// Law values of every lane at the stages it was evaluated at.
pub(crate) struct Sample {
    pub discharge: Vec<f64>,
    pub sensitivity: Vec<f64>,
    pub regime: Vec<f64>,
    pub stages: Vec<[f64; 2]>,
}

/// One `[n]` tensor of `values`.
pub(crate) fn column(values: Vec<f64>) -> TestResult<Tensor<CpuRuntime>> {
    let n = values.len();
    Ok(Tensor::<CpuRuntime>::from_slice(
        &values,
        &[n],
        &CpuDevice::new(),
    )?)
}

/// One for `true`, zero for `false`.
pub(crate) fn flag(value: bool) -> f64 {
    f64::from(u8::from(value))
}

/// Every lane's law, after `shift = (side, sign)` moves one endpoint by `sign × step`.
pub(crate) fn evaluate(
    structures: &CouplingStructures<CpuRuntime>,
    lanes: &[Lane],
    shift: Option<(usize, f64)>,
) -> TestResult<Sample> {
    let stages: Vec<[f64; 2]> = lanes
        .iter()
        .map(|lane| {
            let mut stage = lane.stage;
            if let Some((side, sign)) = shift {
                stage[side] += sign * lane.step;
            }
            stage
        })
        .collect();
    let side = |k: usize| column(stages.iter().map(|s| s[k]).collect());
    let wet = |k: usize| column(lanes.iter().map(|l| flag(l.wet[k])).collect());
    let bed = column(vec![BED; lanes.len()])?;
    let law = structures.evaluate_heads(&side(0)?, &side(1)?, &bed, &bed, &wet(0)?, &wet(1)?)?;
    Ok(Sample {
        discharge: law.discharge.try_to_vec()?,
        sensitivity: law.head_sensitivity.try_to_vec()?,
        regime: law.regime.try_to_vec()?,
        stages,
    })
}

/// Checks `|Q(H+δ) - Q(H-δ)| / 2δ <= s |Q| (1 + SLACK)` at either endpoint of every lane.
///
/// `s` must be finite and nonnegative, and zero where `Q` is zero. Returns the number of lanes
/// that carry water.
pub(crate) fn check_bound(
    structures: &CouplingStructures<CpuRuntime>,
    lanes: &[Lane],
) -> TestResult<usize> {
    let Sample {
        discharge,
        sensitivity,
        ..
    } = evaluate(structures, lanes, None)?;
    let mut flowing = 0;
    for (index, (&q, &s)) in discharge.iter().zip(&sensitivity).enumerate() {
        assert!(s.is_finite() && s >= 0.0, "lane {index}: s={s}");
        if q == 0.0 {
            assert_eq!(s, 0.0, "lane {index}: no flow, so no sensitivity");
        } else {
            flowing += 1;
        }
    }
    for side in 0..2 {
        let up = evaluate(structures, lanes, Some((side, 1.0)))?;
        let down = evaluate(structures, lanes, Some((side, -1.0)))?;
        for index in 0..lanes.len() {
            let width = up.stages[index][side] - down.stages[index][side];
            let difference = (up.discharge[index] - down.discharge[index]).abs() / width;
            let bound = sensitivity[index] * discharge[index].abs() * (1.0 + SLACK);
            assert!(
                difference <= bound,
                "lane {index} side {side}: stages={:?}, difference={difference}, bound={bound}",
                lanes[index].stage
            );
        }
    }
    Ok(flowing)
}
